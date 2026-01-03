# %%
import pickle
import d3rlpy
from d3rlpy.ope import FQE, DiscreteFQE, FQEConfig
from d3rlpy.dataset import MDPDataset
from utils_RD import create_Ohio_dataset, create_MIMIC3_dataset
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.abspath(os.path.dirname(__file__)))

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--method', type=str, default='OneStep', help='SEEK / VSLASSO / SFS / RewardOnly / OneStep / All / VAE (All and VAE are only for MIMIC3)')
parser.add_argument('--orl-method', type=str, default='IQL', help='CQL / IQL / BCQ / NSAC')
parser.add_argument('--data', type=str, default='MIMIC3', help='Ohio / MIMIC3')
parser.add_argument('--include-noise', type=bool, default=False)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

METHOD = args.method
OffineRLMethod = args.orl_method
DATA = args.data
INCLUDE_NOISE_VAR = args.include_noise
N_CRITIC = 2

d3rlpy.seed(args.seed)

if DATA == "Ohio":
    if METHOD == "SEEK":
        var_indices=list(range(12))
    if METHOD == "SFS":
        var_indices=[1, 3, 5, 6, 8, 10, 11]
    if METHOD == "VSLASSO":
        var_indices=list(range(22))
    if METHOD == "RewardOnly":
        var_indices=[9]
    if METHOD == "OneStep":
        var_indices=list(range(22))
elif DATA == "MIMIC3":
    if METHOD != "VAE":
        if METHOD != "ALL":
            if INCLUDE_NOISE_VAR:
                VARS_INDICES_PATH = '{}-Result/{}_{}_manual.pkl'.format(DATA, METHOD, DATA)
            else:
                VARS_INDICES_PATH = '{}-Result/{}_{}.pkl'.format(DATA, METHOD, DATA)
            with open(VARS_INDICES_PATH, 'rb') as handle:
                var_indices = pickle.load(handle)
        else:
            var_indices = list(range(47))
        print("{} selects variables: ".format(METHOD), var_indices)

# %% Create dataset
if DATA == "Ohio":
    if OffineRLMethod == "IQL":
        dataset = create_Ohio_dataset(var_indices, action_continuous=True)
    else:
        dataset = create_Ohio_dataset(var_indices)
if DATA == "MIMIC3":
    if OffineRLMethod == "IQL":
        if METHOD != "VAE":
            dataset = create_MIMIC3_dataset(var_indices, include_noise_var=args.include_noise, action_continuous=True)
        else:
            VAE_DATA_FILE_NAME = 'MIMIC3-Result/VAE_MIMIC3.pkl'
            with open(VAE_DATA_FILE_NAME, 'rb') as handle:
                dataset = pickle.load(handle)
            dataset = MDPDataset(
                observations=dataset[0],
                actions=dataset[1] + np.random.normal(size=np.size(dataset[1]), scale=1e-3),
                rewards=dataset[2],
                terminals=dataset[3],
            ) 
    else:
        dataset = create_MIMIC3_dataset(var_indices, include_noise_var=args.include_noise)
with open('{}-Result/{}_{}_dataset.pkl'.format(DATA, METHOD, OffineRLMethod), 'wb') as handle:
    pickle.dump(dataset, handle)

# %% Learning
from d3rlpy.algos import DiscreteCQLConfig, DiscreteSACConfig, IQLConfig

if OffineRLMethod == "CQL":
    orl = DiscreteCQLConfig(n_critics=N_CRITIC).create(device='cpu')
if OffineRLMethod == "NSAC":
    orl = DiscreteSACConfig(n_critics=10).create(device='cpu')
if OffineRLMethod == "IQL":
    orl = IQLConfig(n_critics=N_CRITIC).create(device='cpu')

orl.fit(
    dataset,
    n_steps=100000,
    n_steps_per_epoch=10000,
    save_interval=10000,
    experiment_name="{}_{}_{}".format(DATA, METHOD, OffineRLMethod),
)
orl.save('{}-Result/{}_{}.d3'.format(DATA, METHOD, OffineRLMethod))

# %% Evaluation
with open('{}-Result/{}_{}_dataset.pkl'.format(DATA, METHOD, OffineRLMethod), 'rb') as handle:
    dataset = pickle.load(handle)
orl = d3rlpy.load_learnable('{}-Result/{}_{}.d3'.format(DATA, METHOD, OffineRLMethod))

if OffineRLMethod == "CQL":
    fqe = DiscreteFQE(algo=orl, config=FQEConfig(n_critics=2))
else:
    fqe = FQE(algo=orl, config=FQEConfig(n_critics=N_CRITIC, batch_size=256, target_update_interval=100, learning_rate=0.0003))

fqe.fit(
    dataset,
    n_steps=100000,
    evaluators={
        "init_value": d3rlpy.metrics.InitialStateValueEstimationEvaluator(),
        "soft_opc": d3rlpy.metrics.SoftOPCEvaluator(return_threshold=-300),
    },
    save_interval=10000,
    experiment_name="FQE_{}_{}_{}".format(DATA, METHOD, OffineRLMethod),
)

# %%
