# %% Prequsisite
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--method', type=str, default='SEEK', help='SEEK / VSLASSO / SFS / RewardOnly / OneStep / VAE')
parser.add_argument('--include-noise', type=bool, default=False)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

METHOD = args.method
SEED = args.seed
NROW = min(300000, 240756)
INCLUDE_NOISE_VAR = args.include_noise
if INCLUDE_NOISE_VAR:
    FILE_NAME = 'preprocess_mimic3_manual.pkl'
    P_AR = 10
else:
    FILE_NAME = 'preprocess_mimic3.pkl'
    P_AR = 0
ID_FILE_NAME = "mimic3_id.pkl"

# %% Data processing
import pandas as pd
df = pd.read_csv("MIMIC3-Result/sepsis_processed_state_action.csv", nrows=NROW)
df.head()
file1 = open('MIMIC3-Result/state_features.txt', 'r')
Lines = file1.readlines()
state_names = [line.strip() for line in Lines]

import numpy as np
from sklearn.preprocessing import normalize
from utils_RD import generate_ar1

df.sort_values(by=['icustayid', 'bloc'], inplace=True)


df['state'] = df.apply(lambda row: np.array([row[x] for x in state_names]).flatten(), axis=1)
if INCLUDE_NOISE_VAR:
    null_var_list = []
    for val in df['icustayid'].value_counts(sort=False):
        null_var = generate_ar1(val, P_AR)
        null_var_list.append(null_var)
    null_var = np.vstack(null_var_list)
    df['null_state'] = [row for row in null_var]
    df['state'] = df.apply(lambda row: np.concatenate([row['state'], row['null_state']]), axis=1)
df['action'] = df['iv_input'] * df['vaso_input'].nunique() + df['vaso_input']    # relabelled action
df['reward'] = -df[['SOFA']]

# convert dataset for RL 
df = df[['icustayid', 'bloc', 'state', 'action', 'reward']]
df['next_state'] = df.groupby('icustayid')['state'].shift(-1, fill_value=np.zeros(len(state_names)+P_AR))
df['done'] = df.groupby('icustayid')['bloc'].transform(lambda x: x == x.max())
dataset = df[['state', 'action', 'reward', 'next_state', 'done']].copy()
dataset.rename(columns={'state': 's1', 'action': 'a', 'reward': 'r', 'next_state': 's2', 'done': 'done'}, inplace=True)

dataset['r'] = (dataset['r'] - dataset['r'].mean()) / dataset['r'].std()

id = df[['icustayid']]

# save to disk
id.to_pickle('MIMIC3-Result/{}'.format(ID_FILE_NAME))
dataset.to_pickle('MIMIC3-Result/{}'.format(FILE_NAME))

# %% load dataset from disk

import pandas as pd
import numpy as np
from scipy.stats import rankdata

dataset = pd.read_pickle('MIMIC3-Result/{}'.format(FILE_NAME))
id = pd.read_pickle('MIMIC3-Result/{}'.format(ID_FILE_NAME))

# %% Beta-Mixing Estimation

from BetaMixingEst import determine_best_K

dataset_s = np.array([dataset["s1"][i] for i in range(len(dataset))])
if METHOD in ['SEEK', 'VSLASSO', 'RewardOnly', 'OneStep']:
    dataset_done = np.array([dataset["done"][i] for i in range(len(dataset))])
    dataset_for_K_selection = [dataset_s, dataset_done]
    K_flag = determine_best_K(dataset_for_K_selection)
else:
    K_flag = 0

# %% Sample division
np.random.seed(SEED)

id = pd.read_pickle('MIMIC3-Result/{}'.format(ID_FILE_NAME))

num_splits = K_flag
id['order'] = id.groupby('icustayid').cumcount() + 1
id['fold_id'] = 0
sub_df_list = []
for i in id['icustayid'].unique():
    sub_df = id[id['icustayid'] == i]
    random_id = np.random.permutation(num_splits)
    for ith in range(num_splits):
        sub_df.iloc[ith::num_splits, sub_df.columns.get_loc('fold_id')] = random_id[ith]
    sub_df_list.append(sub_df)
id = pd.concat(sub_df_list, axis=0)
sample_list = []
for ith in range(num_splits):
    sample_i = dataset[id["fold_id"] == ith]
    sample_list.append(sample_i)

# %% Variable selection

from datetime import datetime
import knockpy as kpy
import pycasso
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import collections
from glmnet import ElasticNet
from sklearn.preprocessing import StandardScaler
from AutoEncoder import Autoencoder, customLoss, weights_init_uniform_rule
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import trange
import pickle

num_splits = K_flag

### SEEK parameter ###
q = 0.1                 # FDR level
# reg_method = "lasso"
reg_method = "randomforest"
### SFS parameter ###
gam = 0.99  
### VAE parameter ###
H = 50
H2 = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_EPOCHS = 20
LATENT_DIM = 5

G_set_all = []
tstart = datetime.now()

if METHOD != "SFS" and METHOD != "VAE":
    for ith in range(num_splits):
        sample_i = sample_list[ith]
        action_space = np.unique(sample_i["a"])
        p = dataset_s.shape[1]
        if METHOD == "SEEK" or METHOD == "RewardOnly" or METHOD == "OneStep":
            Ghat = []
            knockoff_2_mats = {}
            knockoff_2_mats_label = {}
            beta_mat = np.empty(shape=(len(action_space), 2 * p + 1), dtype=np.float64)
            flag = -1
            while flag == -1 or len(ith_Ghat) > 0:

                for j in range(len(action_space)):
                    sample_i_aj = sample_i[sample_i["a"] == action_space[j]]
                    if len(sample_i_aj) <= 2:
                        continue

                    X_train = np.array([x for x in sample_i_aj["s1"]])
                    n = len(X_train)

                    # normalization
                    for jj in range(p):
                        X_train[:, jj] = X_train[:, jj] - np.mean(X_train[:, jj])
                    std = np.sqrt(np.sum(X_train * X_train, axis=0)) / np.sqrt(n)

                    for jj in range(p):
                        if std[jj] != 0:
                            X_train[:, jj] = X_train[:, jj] / std[jj]

                    # store knockoffs
                    if knockoff_2_mats_label.get(str(j)) == None:
                        # Generate second-order knockoffs by sdp
                        Xk_train_g = kpy.knockoffs.GaussianSampler(
                            X=X_train, method="sdp"
                        ).sample_knockoffs()

                        knockoff_2_mats[str(j)] = Xk_train_g
                        knockoff_2_mats_label[str(j)] = j

                    Xk_train_g = knockoff_2_mats[str(j)]
                    Xmat = np.concatenate((X_train, Xk_train_g), 1)
                    if flag == -1:
                        if METHOD == "SEEK" or METHOD == "RewardOnly":
                            yval = sample_i_aj["r"]
                        else:
                            yval = np.hstack([sample_i_aj["r"].to_numpy().reshape(-1, 1), 
                                              np.array([x for x in sample_i_aj["s2"]])])
                            ith_Ghat = list(range(p))
                    else:
                        yval = np.array([x for x in sample_i_aj["s2"]])[
                            :, ith_Ghat
                        ]  # Ghat[flag]]

                    if reg_method not in ['linear', 'logistic', 'lasso', 'randomforest']:
                        print("reg_method must be one of ['linear', 'logistic', 'lasso', 'randomforest']")
                        raise ValueError

                    if reg_method == "linear":
                        # linear
                        dimp = (
                            np.shape(Xmat)[1] + 1
                        )  # plus one because LinearRegression adds an intercept term
                        X_with_intercept = np.empty(
                            shape=(len(yval), dimp), dtype=np.float64
                        )
                        X_with_intercept[:, 0] = 1
                        X_with_intercept[:, 1:dimp] = Xmat
                        if flag == -1:
                            beta_mat[j] = (
                                (
                                    np.linalg.inv(
                                        X_with_intercept.T @ X_with_intercept
                                        + 0 * np.eye(dimp)
                                    )
                                ).dot(X_with_intercept.T)
                            ).dot(yval)
                        else:
                            beta_mat_states = np.empty(
                                shape=(len(ith_Ghat), 2 * p + 1), dtype=np.float64
                            )

                            for ith_state in range(len(ith_Ghat)):
                                yval = np.array([x for x in sample_i_aj["s2"]])[
                                    :, ith_Ghat[ith_state]
                                ]
                                beta_mat_states[ith_state] = (
                                    (
                                        np.linalg.inv(
                                            X_with_intercept.T @ X_with_intercept
                                            + 0 * np.eye(dimp)
                                        )
                                    ).dot(X_with_intercept.T)
                                ).dot(yval)

                            beta_mat[j] = np.max(np.abs(beta_mat_states), 0)

                    if reg_method == "logistic":
                        # linear
                        if flag == -1:
                            if len(yval.unique()) == 1:
                                beta_mat[j] = np.array([0] * (2 * p + 1))
                            else:
                                beta_mat[j] = np.array(
                                    [0]
                                    + LogisticRegression(random_state=SEED)
                                    .fit(Xmat, yval)
                                    .coef_[0]
                                    .tolist()
                                ).reshape(1, -1)
                        else:
                            dimp = np.shape(Xmat)[1]
                            lambda_list = np.exp(
                                np.arange(-20, -8, 0.1)
                            )  # *np.sqrt(np.log(p))#/np.sqrt(n) #np.exp(np.arange(-10,-3,0.1)) #* np.sqrt(np.log(p)) #/ np.sqrt(n) #* np.sqrt(np.log(p)) / np.sqrt(n)  #np.exp(np.arange(-10,1,0.1)) * np.sqrt(np.log(p)) / np.sqrt(n)
                            beta_mat_states = np.empty(
                                shape=(len(ith_Ghat), 2 * p + 1), dtype=np.float64
                            )
                            for ith_state in range(len(ith_Ghat)):
                                yval = np.array([x for x in sample_i_aj["s2"]])[
                                    :, ith_Ghat[ith_state]
                                ]
                                lasso = pycasso.Solver(
                                    Xmat,
                                    yval - np.mean(yval),
                                    penalty="l1",
                                    lambdas=lambda_list,
                                )
                                lasso.train()

                                BIC = np.zeros(len(lambda_list))
                                for k in range(len(lambda_list)):
                                    BIC[k] = np.sum(
                                        np.square(
                                            yval
                                            - np.mean(yval)
                                            - Xmat @ lasso.coef()["beta"][k]
                                        )
                                    ) + sum(lasso.coef()["beta"][k] != 0) * np.log(n)
                                beta_mat_states[ith_state] = np.array(
                                    [0] + lasso.coef()["beta"][np.argmin(BIC)].tolist()
                                ).reshape(1, -1)
                                print("BIC indx:", np.argmin(BIC))
                            beta_mat[j] = np.max(np.abs(beta_mat_states), 0)

                    if reg_method == "lasso":
                        # lasso
                        if flag == -1:
                            dimp = np.shape(Xmat)[1]
                            lambda_list = np.exp(np.arange(-10, -5, 0.1)) / np.sqrt(n)
                            lasso = pycasso.Solver(
                                Xmat,
                                yval - np.mean(yval),
                                penalty="l1",
                                lambdas=lambda_list,
                            )
                            lasso.train()

                            BIC = np.zeros(len(lambda_list))
                            for k in range(len(lambda_list)):
                                BIC[k] = np.sum(
                                    np.square(
                                        yval
                                        - np.mean(yval)
                                        - Xmat @ lasso.coef()["beta"][k]
                                    )
                                ) + sum(lasso.coef()["beta"][k] != 0) * np.log(n)
                            beta_mat[j] = np.array(
                                [0] + lasso.coef()["beta"][np.argmin(BIC)].tolist()
                            ).reshape(1, -1)
                        else:
                            dimp = np.shape(Xmat)[1]
                            lambda_list = np.exp(np.arange(-10, 5, 0.1))  # /np.sqrt(n)
                            beta_mat_states = np.empty(
                                shape=(len(ith_Ghat), 2 * p + 1), dtype=np.float64
                            )
                            for ith_state in range(len(ith_Ghat)):
                                yval = np.array([x for x in sample_i_aj["s2"]])[
                                    :, ith_Ghat[ith_state]
                                ]
                                lasso = pycasso.Solver(
                                    Xmat,
                                    yval - np.mean(yval),
                                    penalty="l1",
                                    lambdas=lambda_list,
                                )
                                lasso.train()

                                BIC = np.zeros(len(lambda_list))
                                for k in range(len(lambda_list)):
                                    BIC[k] = np.sum(
                                        np.square(
                                            yval
                                            - np.mean(yval)
                                            - Xmat @ lasso.coef()["beta"][k]
                                        )
                                    ) + sum(lasso.coef()["beta"][k] != 0) * np.log(n)
                                beta_mat_states[ith_state] = np.array(
                                    [0] + lasso.coef()["beta"][np.argmin(BIC)].tolist()
                                ).reshape(1, -1)
                                print("BIC indx:", np.argmin(BIC))

                            beta_mat[j] = np.max(np.abs(beta_mat_states), 0)

                    if reg_method == "randomforest":
                        # N_TREES = min(int(n / 10), 200)
                        MIN_SAMPLES_LEAF = 1
                        MAX_DEPTH = 6
                        N_TREES = 50
                        # MAX_DEPTH = None
                        # MIN_SAMPLES_LEAF = 20
                        if METHOD != "OneStep" and flag == -1:
                            if len(yval.unique()) == 1:
                                beta_mat[j] = np.array([0] * (2 * p + 1))
                            else:
                                rf = RandomForestRegressor(
                                    n_estimators=N_TREES,
                                    min_samples_leaf=MIN_SAMPLES_LEAF,
                                    max_depth=MAX_DEPTH,
                                    max_features="sqrt",
                                    random_state=SEED,
                                )
                                rf.fit(Xmat, yval)
                                beta_mat[j] = np.array(
                                    [0] + (rf.feature_importances_).tolist()
                                ).reshape(1, -1)
                        else:
                            beta_mat_states = np.empty(
                                shape=(len(ith_Ghat), 2 * p + 1), dtype=float
                            )
                            for ith_state in range(len(ith_Ghat)):
                                yval = np.array([x for x in sample_i_aj["s2"]])[
                                    :, ith_Ghat[ith_state]
                                ]
                                n_unique = len(np.unique(yval))
                                if n_unique <= 5:
                                    yval = np.round(yval).astype(int)
                                    rf = RandomForestClassifier(
                                        n_estimators=N_TREES,
                                        min_samples_leaf=MIN_SAMPLES_LEAF,
                                        max_depth=MAX_DEPTH,
                                        max_features="sqrt",
                                        random_state=SEED,
                                    )
                                else:
                                    rf = RandomForestRegressor(
                                        n_estimators=N_TREES,
                                        min_samples_leaf=MIN_SAMPLES_LEAF,
                                        max_depth=MAX_DEPTH,
                                        max_features="sqrt",
                                        random_state=SEED,
                                    )
                                rf.fit(Xmat, yval)
                                beta_mat_states[ith_state] = np.array(
                                    [0] + (rf.feature_importances_).tolist()
                                ).reshape(1, -1)
                            # beta_mat[j] = np.max(np.abs(beta_mat_states), 0)
                            beta_mat[j] = np.max(beta_mat_states, 0)

                if reg_method != 'randomforest':
                    max_beta = np.max(np.abs(beta_mat), 0)
                else:
                    max_beta = np.max(beta_mat, 0)

                # adaptive Wi
                rest_list = list(set(range(p)) - set(Ghat))
                Wi = (max_beta[1 : (p + 1)] - max_beta[(p + 1) :])[rest_list]

                # find t
                W_abs = np.sort(np.abs(Wi))
                for i in range(len(W_abs)):
                    tt = W_abs[i]
                    if (np.sum(Wi <= -tt)) / np.sum(Wi >= tt) <= q:
                        break
                tau = tt

                if tau == 0:
                    next_ith_Ghat = []
                else:
                    next_ith_Ghat = [
                        rest_list[i]
                        for i, x in enumerate((Wi >= tau).tolist())
                        if x == True
                    ]

                if METHOD != "OneStep":
                    if flag == -1:
                        print("Knockoff Set on Reward:", next_ith_Ghat)
                    else:
                        print("Knockoff Set on State " + str(ith_Ghat) + " :", next_ith_Ghat)
                else:
                    print("Knockoff Set on All States:", next_ith_Ghat)
                for idx in next_ith_Ghat:
                    if idx not in Ghat:
                        Ghat.append(idx)

                print("tau:", tau)

                flag = flag + 1
                if METHOD == "SEEK":
                    ith_Ghat = next_ith_Ghat
                if METHOD == "RewardOnly" or METHOD == "OneStep":
                    ith_Ghat = []

            G_set_all.append(set(Ghat))
            print("Final Knockoff Set based on Split Sample " + str(ith) + " :", Ghat)
        
        if METHOD == "VSLASSO":
            beta_mat = np.empty(shape=(len(action_space), p + 1), dtype=np.float64)

            for j in range(len(action_space)):

                sample_i_aj = sample_i[sample_i['a'] == action_space[j]]

                X_train = np.array([x for x in sample_i_aj['s1']])
                n=len(X_train) 

                # normalization
                for jj in range(p):
                    X_train[:,jj] = X_train[:,jj] - np.mean(X_train[:,jj])
                std = np.sqrt(np.sum(X_train * X_train, axis=0))/np.sqrt(n)
                Xmat = X_train/std

                yval_mat =  np.concatenate((np.array([x for x in sample_i_aj['s2']]),np.array(sample_i_aj['r']).reshape(-1,1)),1)  
                dimp = np.shape(Xmat)[1]
                lambda_list = np.exp(np.arange(-10, 5, 0.1))
                #*np.sqrt(np.log(p))#/np.sqrt(n) #np.exp(np.arange(-10,-3,0.1)) #* np.sqrt(np.log(p)) #/ np.sqrt(n) #* np.sqrt(np.log(p)) / np.sqrt(n)  #np.exp(np.arange(-10,1,0.1)) * np.sqrt(np.log(p)) / np.sqrt(n) 
                beta_mat_states = np.empty(shape=(p + 1, p + 1), dtype=np.float64)
                for ith_state in range(p + 1):
                    yval =  yval_mat[:,ith_state]
                    lasso = pycasso.Solver(Xmat, yval-np.mean(yval), penalty="l1", lambdas=lambda_list)
                    lasso.train()
                    BIC = np.zeros(len(lambda_list))
                    for k in range(len(lambda_list)):
                        BIC[k] = np.sum(np.square(yval - np.mean(yval) - Xmat @ lasso.coef()['beta'][k])) + \
                                        sum(lasso.coef()['beta'][k]!=0)*np.log(n) 
                    beta_mat_states[ith_state] = np.array([0] + lasso.coef()['beta'][np.argmin(BIC)].tolist()).reshape(1,-1)
                    print('BIC indx:',np.argmin(BIC))
                beta_mat[j] = np.max(np.abs(beta_mat_states),0)

            max_beta = np.max(np.abs(beta_mat),0)[1:] 
            print(max_beta) 
            rest_list=list(range(p))
            Wi = max_beta[rest_list]
            print('Wi', Wi)
            next_ith_Ghat = [rest_list[i] for i, x in enumerate((Wi > 2.5).tolist()) if x == True]

            G_set_all.append(set(next_ith_Ghat))
            print('Final Lasso Set based on Split Sample ' + str(ith) + ' :', next_ith_Ghat)
else:    
    num_act = len(np.unique(dataset['a']))

    train_state = np.array(dataset['s1'].tolist())
    train_state = StandardScaler(copy=True).fit_transform(train_state)
    train_next_state = np.array(dataset['s1'].tolist())
    train_next_state = StandardScaler(copy=True).fit_transform(train_next_state)
    train_reward = np.array(dataset['r'].tolist())
    train_reward = StandardScaler(copy=True).fit_transform(train_reward.reshape(-1, 1)).flatten()
    train_action = np.array(dataset['a'].tolist())
    train_done = np.array(dataset['done'].tolist())

    if METHOD == "VAE":
        D_IN = train_state.shape[1]
        model = Autoencoder(D_IN, H, H2, LATENT_DIM).to(DEVICE)
        x_train_torch = torch.from_numpy(train_state).float().to(DEVICE)
        x_test_torch = torch.from_numpy(train_next_state).float().to(DEVICE)
        train_loader=DataLoader(dataset=x_train_torch, batch_size=256)
        test_loader=DataLoader(dataset=x_test_torch, batch_size=256)
        model.apply(weights_init_uniform_rule)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_mse = customLoss()

        # Training the Model
        def train(epoch):
            model.train()
            train_loss = 0
            for batch_idx, data in enumerate(train_loader):
                data = data.to(DEVICE)
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(data)
                loss = loss_mse(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            if epoch % 10 == 0:
                print('====> Epoch: {} Average loss: {:.4f}'.format(
                    epoch, train_loss / len(train_loader.dataset)))

        for epoch in trange(1, TRAIN_EPOCHS + 1):
            train(epoch)

        train_mu = []
        with torch.no_grad():
            for i, (data) in enumerate(train_loader):
                data = data.to(DEVICE)
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(data)
                train_mu.append(mu)
            train_mu = torch.cat(train_mu, dim=0)
        train_state = train_mu.detach().cpu().numpy()
        print(train_state.shape)

        # test_mu = []
        # with torch.no_grad():
        #     for i, (data) in enumerate(test_loader):
        #         data = data.to(DEVICE)
        #         optimizer.zero_grad()
        #         recon_batch, mu, logvar = model(data)
        #         test_mu.append(mu)
        #     test_mu = torch.cat(test_mu, dim=0)
        # train_next_state = test_mu.detach().cpu().numpy()
        # print(train_next_state.shape)

        VAE_DATA_FILE_NAME = 'MIMIC3-Result/VAE_MIMIC3.pkl' 
        dataset = [train_state, train_action, train_reward.flatten(), train_done.flatten()]
        with open(VAE_DATA_FILE_NAME, 'wb') as handle:
            pickle.dump(dataset, handle)
    elif METHOD == "SFS":
        train_data = [[train_state[train_action==ith_act, :], train_reward[train_action==ith_act], train_next_state[train_action==ith_act, :], train_done[train_action==ith_act]] for ith_act in range(num_act)]

        ## Initialize Q-function to be zero and construct target
        regrs = []
        for ith_act in range(num_act): 
            regr_i = ElasticNet()
            regr_i.fit(train_data[ith_act][0], train_data[ith_act][1])
            regrs.append(regr_i)

        ## FQI
        p = train_state.shape[1]
        for kk in range(10):
            y = list()
            for ith_act in range(num_act): 
                y.append(gam * regrs[ith_act].predict(train_next_state).reshape(-1, 1))
            y = np.hstack(y)
            y = np.max(y, axis=1)
            y = train_reward + (1 - train_done) * y

            regrs = []
            for ith_act in range(num_act): 
                regr_i = ElasticNet()
                regrs.append(regr_i.fit(train_state[train_action==ith_act, :], y[train_action==ith_act, ]))

        beta_mat = np.empty(shape=(num_act, p), dtype=np.float64)   
        for ith_act in range(num_act): 
            beta_mat[ith_act] = regrs[ith_act].coef_

        max_beta = np.max(np.abs(beta_mat),0) 

        vote = [list(range(p))[i] for i, x in enumerate((max_beta > 0).tolist()) if x == True]
speed = (datetime.now() - tstart).seconds / 60

# majority vote
if METHOD != "SFS" and METHOD != "VAE":
    all_list = []
    for iset in G_set_all:
        all_list = all_list + list(iset)
    all_list

    counter = collections.Counter(all_list)

    vote_res = (np.array(list(counter.values())) / len(G_set_all) >= 0.5).tolist()
    vote = []
    for i in range(len(vote_res)):
        if vote_res[i] == True:
            vote.append(list(counter.keys())[i])

    print(
        "K selected:", K_flag, "Estimated Ghat: ", vote, "Time Spent (Minutes): ", speed
    )
elif METHOD == "SFS":
    print('Estimated Ghat: ', vote, 'Time Spent (Minutes): ', speed)
else:
    pass

# %% save state selection results
import pickle
if METHOD != "VAE":
    if INCLUDE_NOISE_VAR:
        RESULT_FILE_NAME = 'MIMIC3-Result/{}_MIMIC3_manual.pkl'.format(METHOD)
    else:
        RESULT_FILE_NAME = 'MIMIC3-Result/{}_MIMIC3.pkl'.format(METHOD)
    if os.path.exists(RESULT_FILE_NAME):
        os.remove(RESULT_FILE_NAME)
    with open(RESULT_FILE_NAME, 'wb') as handle:
        pickle.dump(vote, handle)
    vote