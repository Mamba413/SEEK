from BetaMixingEst import determine_best_K
import numpy as np
import random
import data_generator
import pickle
from datetime import datetime
from tqdm import trange


# %%
def exp_select_K(settings, seed):

    seed = int(seed)

    np.random.seed(seed)
    random.seed(seed)

    p = settings[0]
    n_traj = settings[1]
    ar_cons = settings[2]
    env_name = settings[3]
    init_K = settings[4]
    GMM_num_component = settings[5]
    parametric_form = settings[6]
    BIC_select = settings[7]
    num_splits_method = "BetaMix"
    error_threshold = 0.01

    eps = 0.3
    _mu = 0
    _sigma = 1

    if env_name == "AR1":

        data_gen = data_generator.DataGenerator_AR1(p=p, mu=_mu, sigma=_sigma)

        dataset = data_gen.get_dataset(seed=seed, n=n_traj, eps=eps, ar_cons=ar_cons)

    if env_name == "Mixed":

        data_gen = data_generator.DataGenerator_Toy(p=p, mu=_mu, sigma=_sigma)

        dataset = data_gen.get_dataset(seed=seed, n=n_traj, eps=eps, ar_cons=ar_cons)

    # Determine number of disjoint dataset
    if num_splits_method == "BetaMix":
        ## data preparation:
        dataset_s = np.array([dataset["s1"][i] for i in range(len(dataset))])
        dataset_done = np.array([dataset["done"][i] for i in range(len(dataset))])

        dataset_for_K_selection = [dataset_s, dataset_done]
        best_K = determine_best_K(
            dataset_for_K_selection,
            K_max=init_K,
            n_components=GMM_num_component,
            BIC_select=BIC_select,
            parametric_form=parametric_form,
            error_threshold=error_threshold,
        )
    return best_K


# %% AR and Mixed Environments Settings
env_name = "AR1"
# env_name = "Mixed"
if env_name == "AR1":
    # N_TRAJ_LIST = [10, 20, 40]
    N_TRAJ_LIST = [40]
if env_name == "Mixed":
    # N_TRAJ_LIST = [50, 100, 200]
    N_TRAJ_LIST = [200]


########## DEFAULT settings in paper ##########
DEFAULT_K = 15
DEFAULT_NUM_COMPONENT = 1
DEFAULT_PARAMETRIC_FORM = "exp"
p = 20
ar_cons = 0.9
###############################################


########## Experiment settings ##########
rep_number = 100
INIT_K_LIST = [15, 20, 25, 30, 35, 40]
NUM_COMPONENT_LIST = [1, 3, 5, 7, 9, 11, 13]
PARAMETRIC_FORM_LIST = ["exp", "poly"]
#########################################

# %% INIT_K varies
for n_traj in N_TRAJ_LIST:
    for init_K in INIT_K_LIST:
        settings = [
            p,
            n_traj,
            ar_cons,
            env_name,
            init_K,
            DEFAULT_NUM_COMPONENT,
            DEFAULT_PARAMETRIC_FORM,
            False,
        ]

        rep_res = []
        for seed_i in trange(rep_number):
            res_i = exp_select_K(settings, seed_i)
            rep_res.append(res_i)

        date_time = datetime.fromtimestamp(datetime.timestamp(datetime.now())).strftime(
            "%Y%m%d%H%M%S"
        )

        with open(
            "res/SelectK_{}_NumTraj_{}_InitK_{}_NumComp_{}_{}_{}.pkl".format(
                env_name,
                str(n_traj),
                str(init_K),
                str(DEFAULT_NUM_COMPONENT),
                DEFAULT_PARAMETRIC_FORM,
                date_time,
            ),
            "wb",
        ) as filehandle:
            pickle.dump(rep_res, filehandle)

# %% PARAMETRIC_FORM varies
for n_traj in N_TRAJ_LIST:
    for parametric_form in PARAMETRIC_FORM_LIST:
        print(parametric_form)
        settings = [
            p,
            n_traj,
            ar_cons,
            env_name,
            DEFAULT_K,
            DEFAULT_NUM_COMPONENT,
            parametric_form,
            False,
        ]

        rep_res = []
        for seed_i in trange(rep_number):
            res_i = exp_select_K(settings, seed_i)
            rep_res.append(res_i)

        date_time = datetime.fromtimestamp(datetime.timestamp(datetime.now())).strftime(
            "%Y%m%d%H%M%S"
        )

        with open(
            "res/SelectK_{}_NumTraj_{}_InitK_{}_NumComp_{}_{}_{}.pkl".format(
                env_name,
                str(n_traj),
                str(DEFAULT_K),
                str(DEFAULT_NUM_COMPONENT),
                parametric_form,
                date_time,
            ),
            "wb",
        ) as filehandle:
            pickle.dump(rep_res, filehandle)

# %% NUM_COMPONENT varies
for n_traj in N_TRAJ_LIST:
    for n_comp in NUM_COMPONENT_LIST:
        settings = [
            p,
            n_traj,
            ar_cons,
            env_name,
            DEFAULT_K,
            n_comp,
            DEFAULT_PARAMETRIC_FORM,
            False,
        ]

        rep_res = []
        for seed_i in trange(rep_number):
            res_i = exp_select_K(settings, seed_i)
            rep_res.append(res_i)

        date_time = datetime.fromtimestamp(datetime.timestamp(datetime.now())).strftime(
            "%Y%m%d%H%M%S"
        )

        with open(
            "res/SelectK_{}_NumTraj_{}_InitK_{}_NumComp_{}_{}_{}.pkl".format(
                env_name,
                str(n_traj),
                str(DEFAULT_K),
                str(n_comp),
                DEFAULT_PARAMETRIC_FORM,
                date_time,
            ),
            "wb",
        ) as filehandle:
            pickle.dump(rep_res, filehandle)

# %% select NUM_COMPONENT via BIC
for n_traj in N_TRAJ_LIST:
    for n_comp in NUM_COMPONENT_LIST:
        settings = [
            p,
            n_traj,
            ar_cons,
            env_name,
            DEFAULT_K,
            n_comp,
            DEFAULT_PARAMETRIC_FORM,
            True,
        ]

        rep_res = []
        for seed_i in trange(rep_number):
            res_i = exp_select_K(settings, seed_i)
            rep_res.append(res_i)

        date_time = datetime.fromtimestamp(datetime.timestamp(datetime.now())).strftime(
            "%Y%m%d%H%M%S"
        )

        with open(
            "res/SelectK_{}_AutoNumTraj_{}_InitK_{}_NumComp_{}_{}_{}.pkl".format(
                env_name,
                str(n_traj),
                str(DEFAULT_K),
                str(n_comp),
                DEFAULT_PARAMETRIC_FORM,
                date_time,
            ),
            "wb",
        ) as filehandle:
            pickle.dump(rep_res, filehandle)