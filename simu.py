# %%
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import gym
import numpy as np
import data_generator
import sys
from io import StringIO

import utils_batchRL

import scipy

from utils import *
from collections import deque
import torch
import argparse
import glob
import random
from copy import deepcopy
import math
from datetime import datetime

import collections

import os
import pickle
import random
from multiprocessing import Pool
from datetime import datetime
import argparse
from tqdm import tqdm
from functools import partial

import pycasso
from glmnet import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

import numpy as np

# import tensorflow as tf

# from DeepKnockoffs import KnockoffMachine
# from DeepKnockoffs import GaussianKnockoffs
import knockpy as kpy
from knockpy.knockoff_filter import KnockoffFilter
import json

from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal

from stable_baselines3 import DQN, PPO
from sb3_contrib import TRPO

import warnings

warnings.filterwarnings("ignore")

VERBOSE = False


# %%
def Beta_Mixing_Est_Gaussian(dataset, a, sam_size=10000, n_components=5):
    """
    Estimate Beta-Mixing by Mix Gaussian Method
    @para:
    dataset: input data
    a: blocks of interest
    sam_size: sampling size in uniform distribution to calculate integral
    """
    dataset_s = dataset[0]
    dataset_done = dataset[1]

    valid_list = np.array(
        [
            sum(dataset_done[int(ith) : int(ith + a)])
            for ith in range(len(dataset_s) - a)
        ]
    )

    ## Form (S_t, S_{t+a})
    sample_joint = np.array(
        [
            dataset_s[int(ith)].tolist() + dataset_s[int(ith + a)].tolist()
            for ith in range(len(dataset_s) - a)
        ]
    )

    ## Form (S_t)
    sample_single = np.array(
        [dataset_s[int(ith)].tolist() for ith in range(len(dataset_s) - a)]
    )

    ## Delete jump cases
    sample_joint = sample_joint[np.where(valid_list == 0)[0].tolist()]
    sample_single = sample_single[np.where(valid_list == 0)[0].tolist()]

    joint_dim = sample_joint.shape[1]
    if joint_dim >= 2000:
        # cov_type = 'diag'
        cov_type = "tied"
    elif joint_dim >= 500:
        cov_type = "tied"
    else:
        cov_type = "full"
    if joint_dim >= 2000:
        N_INIT = 1
        MAX_ITER = 300
    else:
        N_INIT = 1
        MAX_ITER = 100
    ## Fit Gaussian Density
    den_single = GaussianMixture(
        n_components=n_components,
        random_state=0,
        covariance_type=cov_type,
        init_params='k-means++',
        n_init=N_INIT,
        max_iter=MAX_ITER,
    ).fit(sample_single)
    den_joint = GaussianMixture(
        n_components=n_components,
        random_state=0,
        covariance_type=cov_type,
        init_params='k-means++',
        n_init=N_INIT,
        max_iter=MAX_ITER,
    ).fit(sample_joint)

    # Sampling State from Reference Gaussian Distribution
    smp_S_1 = den_joint.sample(sam_size)[
        0
    ]  # np.random.multivariate_normal(den_joint.means_[0], sigma_1, sam_size)
    smp_S_2 = np.concatenate(
        (den_single.sample(sam_size)[0], den_single.sample(sam_size)[0]), 1
    )  # np.random.multivariate_normal(den_joint.means_[0], sigma_2, sam_size)
    smp_S = np.concatenate((smp_S_1, smp_S_2), 0)

    density_joint = 0
    density_single_1 = 0
    density_single_2 = 0
    log_den_offset = np.nan
    for ith_mix in range(n_components):
        ## Get Covariance Matrix
        if cov_type == "full":
            sigma_2 = den_single.covariances_[ith_mix]
            sigma_1 = den_joint.covariances_[ith_mix]
        elif cov_type == "tied":
            sigma_2 = den_single.covariances_
            sigma_1 = den_joint.covariances_
        elif cov_type == "diag":
            sigma_2 = np.diag(den_single.covariances_[ith_mix])
            sigma_1 = np.diag(den_joint.covariances_[ith_mix])

        # Get different pdf
        Gaus_pdf_joint = multivariate_normal(
            mean=den_joint.means_[ith_mix], cov=sigma_1
        )

        Gaus_pdf_single = multivariate_normal(
            mean=den_single.means_[ith_mix], cov=sigma_2
        )

        if joint_dim < 1000:
            density_joint = (
                density_joint + Gaus_pdf_joint.pdf(smp_S) * den_joint.weights_[ith_mix]
            )

            density_single_1 = (
                density_single_1
                + Gaus_pdf_single.pdf(smp_S[:, :(p)]) * den_single.weights_[ith_mix]
            )
            density_single_2 = (
                density_single_2
                + Gaus_pdf_single.pdf(smp_S[:, (p):]) * den_single.weights_[ith_mix]
            )
        else:
            joint_log_den = Gaus_pdf_joint.logpdf(smp_S)
            if log_den_offset is np.nan:
                log_den_offset = (-20) - np.mean(joint_log_den)
            density_joint = (
                density_joint
                + np.exp(joint_log_den + log_den_offset) * den_joint.weights_[ith_mix]
            )

            density_single_1 = (
                density_single_1
                + np.exp(Gaus_pdf_single.logpdf(smp_S[:, :(p)]) + (log_den_offset / 2))
                * den_single.weights_[ith_mix]
            )
            density_single_2 = (
                density_single_2
                + np.exp(Gaus_pdf_single.logpdf(smp_S[:, (p):]) + (log_den_offset / 2))
                * den_single.weights_[ith_mix]
            )

    beta_bnd = (density_joint - density_single_1 * density_single_2) / (
        density_joint + density_single_1 * density_single_2
    )
    beta_bnd[np.isnan(beta_bnd)] = 0.0
    beta_bnd = np.mean(np.abs(beta_bnd))

    # print('Beta Upper Bound: ', beta_bnd)

    ## Calculate epsilon

    epsilon = len(dataset[0]) / a * beta_bnd

    # print('Epsilon: ', epsilon)

    return beta_bnd, epsilon


# %%
def exp_knock(settings, seed):

    seed = int(seed)

    np.random.seed(seed)
    random.seed(seed)

    p = settings[0]
    n_traj = settings[1]
    ar_cons = settings[2]
    env_name = settings[3]
    METHOD = settings[4]

    q = 0.1
    K_max = 15
    n_components = 1
    if env_name == "AR1" or env_name == "Mixed":
        eval_episode = 1000
        reg_method = "lasso"
        num_splits_method = "BetaMix"
    elif env_name in ["CartPole-v0", "LunarLander-v2", "MountainCar-v0", "Acrobot-v1"]:
        eval_episode = 100
        reg_method = "randomforest"
        num_splits_method = "Fix"

    POLICY_EVALUATION = True
    if env_name == "LunarLander-v2":
        POLICY_EVALUATION = False

    eps = 0.3
    _mu = 0
    _sigma = 1

    if env_name == "CartPole-v0":
        continuous_reward = False
        opt_policy = torch.load("trained_models/opt_policy_v0.pth")

    if env_name == "Acrobot-v1":
        num_actions = 3
        continuous_reward = False
        action_space = [i for i in range(num_actions)]
        # model = DQN.load("logs/dqn/Acrobot-v1_1/acrobot-v1")
        model = PPO.load("logs/ppo/Acrobot-v1_1/acrobot-v1")
        # model = TRPO.load("logs/trpo/Acrobot-v1_1/acrobot-v1")
        def opt_policy(state, epsilon=0.0):
            if np.random.random() < epsilon:
                action = np.random.choice(action_space)  
            else:
                action, _ = model.predict(state, deterministic=True)
            return [action]
    
    if env_name == "LunarLander-v2":
        continuous_reward = False
        num_actions = 4
        action_space = [i for i in range(num_actions)]
        
        model = DQN.load("logs/dqn/LunarLander-v2_1/lunarLander-v2.zip")
        # model = PPO.load("logs/ppo/LunarLander-v2_1/lunarLander-v2")
        # opt_policy_LunarLander = tf.keras.models.load_model("trained_models/d3qn_model")

        def opt_policy(observation, epsilon=0):
            if np.random.random() < epsilon:
                action = np.random.choice(action_space)
            else:
                action, _ = model.predict(observation, deterministic=True)
                # state = np.array([observation])
                # _, actions = opt_policy_LunarLander(state)
                # action = tf.math.argmax(actions, axis=1).numpy()[0]

            return [action]

    data_path = "data/{}_N{}_p{}_ar{}_seed{}.pkl".format(env_name, str(n_traj), str(p), str(ar_cons), str(seed))
    if os.path.exists(data_path):
        with open(data_path, "rb") as filehandle:
            dataset = pickle.load(filehandle)
    else:
        if env_name == "CartPole-v0":
            data_gen = data_generator.DataGenerator_gym(p=p, mu=_mu, sigma=_sigma)

            dataset = data_gen.get_dataset(
                seed=seed,
                n=n_traj,
                opt_policy=opt_policy,
                env_name=env_name,
                continuous_reward=continuous_reward,
                eps=eps,
                ar_cons=ar_cons,
            )

        if env_name == "Acrobot-v1":
            data_gen = data_generator.DataGenerator_gym(p=p, mu=_mu, sigma=_sigma)

            dataset = data_gen.get_dataset(
                seed=seed,
                n=n_traj,
                opt_policy=opt_policy,
                env_name=env_name,
                continuous_reward=continuous_reward,
                eps=eps,
                ar_cons=ar_cons,
            )

        if env_name == "MountainCar-v0":
            data_gen = data_generator.DataGenerator_gym(p=p, mu=_mu, sigma=_sigma)

            dataset = data_gen.get_dataset(
                seed=seed,
                n=n_traj,
                opt_policy=opt_policy,
                env_name=env_name,
                continuous_reward=continuous_reward,
                eps=eps,
                ar_cons=ar_cons,
            )

        if env_name == "LunarLander-v2":
            data_gen = data_generator.DataGenerator_gym(p=p, mu=_mu, sigma=_sigma)

            dataset = data_gen.get_dataset(
                seed=seed,
                n=n_traj,
                opt_policy=opt_policy,
                env_name=env_name,
                continuous_reward=continuous_reward,
                eps=eps,
                ar_cons=ar_cons,
            )

        if env_name == "AR1":

            data_gen = data_generator.DataGenerator_AR1(p=p, mu=_mu, sigma=_sigma)

            dataset = data_gen.get_dataset(seed=seed, n=n_traj, eps=eps, ar_cons=ar_cons)

        if env_name == "Mixed":

            data_gen = data_generator.DataGenerator_Toy(p=p, mu=_mu, sigma=_sigma)

            dataset = data_gen.get_dataset(seed=seed, n=n_traj, eps=eps, ar_cons=ar_cons)

        with open(data_path, "wb") as filehandle:
            pickle.dump(dataset, filehandle)
    # Determine number of disjoint dataset
    if num_splits_method == "BetaMix" and method not in ["SFS", 'DRL']:
        ## Beta-Mixing Estimation
        dataset_s = np.array([dataset["s1"][i] for i in range(len(dataset))])
        dataset_done = np.array([dataset["done"][i] for i in range(len(dataset))])

        dataset_for_K_selection = [dataset_s, dataset_done]

        est_betas, est_eps = [], []
        for ith_K in range(1, K_max + 1):
            ith_beta, ith_eps = Beta_Mixing_Est_Gaussian(
                dataset_for_K_selection,
                ith_K,
                sam_size=10000,
                n_components=n_components,
            )
            est_betas.append(ith_beta)
            est_eps.append(ith_eps)

        ## Beta-Mixing Debiasing
        K_range = range(1, K_max + 1)

        c_0 = np.min(np.array(est_betas))
        fitted_beta_para = np.polyfit(
            K_range,
            np.log(np.array(est_betas) - c_0 + 1e-10),
            1,
            w=np.array(est_betas) ** 2,
        )
        a_0 = fitted_beta_para[1]
        b_0 = fitted_beta_para[0]

        def fun_obj(paras):
            return np.mean(
                (np.array(est_betas) - paras[0] - paras[1] * np.exp(paras[2] * K_range))
                ** 2
            )

        est_para = scipy.optimize.minimize(fun_obj, x0=[c_0, a_0, b_0]).x
        fitted_a = est_para[1]
        fitted_b = est_para[2]

        K_flag = 1
        error = fitted_a * np.exp(fitted_b * K_flag) * len(dataset) / K_flag
        error_threshold = 0.01
        while error > error_threshold and K_flag <= 5 * np.log(n_traj * p):
            K_flag = K_flag + 1
            error = fitted_a * np.exp(fitted_b * K_flag) * len(dataset) / K_flag

        print("best K =", K_flag)

        num_splits = K_flag
    elif method in ["SFS", "DRL"]:
        num_splits = 1
    else:
        num_splits = int(np.log(len(dataset)))

    # variable selection based on knockoff
    tstart = datetime.now()
    G_set_all = []
    for ith in range(num_splits):

        sample_i = dataset.iloc[ith::num_splits]

        action_space = np.unique(sample_i["a"])

        if METHOD == "SEEK" or METHOD == "RewardOnly" or METHOD == "OneStep":
            Ghat = []

            knockoff_2_mats = {}
            knockoff_2_mats_label = {}

            beta_mat = np.zeros(shape=(len(action_space), 2 * p + 1), dtype=float)

            flag = -1

            while flag == -1 or len(ith_Ghat) > 0:

                for j in range(len(action_space)):

                    sample_i_aj = sample_i[sample_i["a"] == action_space[j]]

                    X_train = np.array([x for x in sample_i_aj["s1"]])
                    n = len(X_train)
                    if n <= 1:
                        continue

                    # normalization

                    for jj in range(p):
                        X_train[:, jj] = X_train[:, jj] - np.mean(X_train[:, jj])
                    std = np.sqrt(np.sum(X_train * X_train, axis=0)) / np.sqrt(n)

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
                        ]  # Ghat[flag]

                    if reg_method == "linear":
                        # linear
                        dimp = (
                            np.shape(Xmat)[1] + 1
                        )  # plus one because LinearRegression adds an intercept term
                        X_with_intercept = np.empty(
                            shape=(len(yval), dimp), dtype=float
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
                                shape=(len(ith_Ghat), 2 * p + 1), dtype=float
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
                        if METHOD != "OneStep" and flag == -1:
                            if len(yval.unique()) == 1:
                                beta_mat[j] = np.array([0] * (2 * p + 1))
                            else:
                                beta_mat[j] = np.array(
                                    [0]
                                    + LogisticRegression(random_state=0)
                                    .fit(Xmat, yval)
                                    .coef_[0]
                                    .tolist()
                                ).reshape(1, -1)
                        else:
                            dimp = np.shape(Xmat)[1]
                            lambda_list = np.exp(
                                np.arange(-20, -8, 0.1)
                            )
                            beta_mat_states = np.empty(
                                shape=(len(ith_Ghat), 2 * p + 1), dtype=float
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
                                if VERBOSE:
                                    print("BIC index:", np.argmin(BIC))
                            beta_mat[j] = np.max(np.abs(beta_mat_states), 0)

                    if reg_method == "lasso":
                        ################################################################
                        original_stdout = sys.stdout
                        sys.stdout = StringIO()   ### avoid lasso.train() print useless information in simulation

                        if METHOD != "OneStep" and flag == -1:
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
                                shape=(len(ith_Ghat), 2 * p + 1), dtype=float
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
                                if VERBOSE:
                                    print("BIC index:", np.argmin(BIC))

                            beta_mat[j] = np.max(np.abs(beta_mat_states), 0)
                        sys.stdout = original_stdout
                        ################################################################

                    if reg_method == "randomforest":
                        if METHOD != "OneStep" and flag == -1:
                            if len(yval.unique()) == 1:
                                beta_mat[j] = np.array([0] * (2 * p + 1))
                            else:
                                rf = RandomForestRegressor(
                                    n_estimators=max(int(n / 30), 10),
                                    max_depth=4,
                                    max_features="sqrt",
                                    random_state=1,
                                )  # (n_estimators=max(100,int(n/30)))
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
                                rf = RandomForestRegressor(
                                    n_estimators=max(int(n / 30), 10),
                                    max_depth=4,
                                    max_features="sqrt",
                                    random_state=1,
                                )
                                rf.fit(Xmat, yval)
                                beta_mat_states[ith_state] = np.array(
                                    [0] + (rf.feature_importances_).tolist()
                                ).reshape(1, -1)

                            beta_mat[j] = np.max(np.abs(beta_mat_states), 0)

                max_beta = np.max(np.abs(beta_mat), 0)

                # adaptive Wi
                rest_list = list(set(range(p)) - set(Ghat))
                Wi = (max_beta[1 : (p + 1)] - max_beta[(p + 1) :])[rest_list]

                # find t in knockoff
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
                print("tau:", tau)

                if METHOD != "OneStep":
                    if flag == -1:
                        print("Knockoff Set on Reward:", next_ith_Ghat)
                    else:
                        print("Knockoff Set on State " + str(ith_Ghat) + " :", next_ith_Ghat)
                else:
                    print("Knockoff Set on All States:", next_ith_Ghat)

                ## For next update:
                for idx in next_ith_Ghat:
                    if idx not in Ghat:
                        Ghat.append(idx)

                flag = flag + 1
                if METHOD == "SEEK":
                    ith_Ghat = next_ith_Ghat
                else:
                    ith_Ghat = []

            G_set_all.append(set(Ghat))
            print("Final Knockoff Set based on Split Sample ".format(METHOD) + str(ith) + " :", Ghat)

        if METHOD == "VS":
            beta_mat = np.empty(shape=(len(action_space), p + 1), dtype=np.float64)

            for j in range(len(action_space)):
                sample_i_aj = sample_i[sample_i["a"] == action_space[j]]
                X_train = np.array([x for x in sample_i_aj["s1"]])
                n = len(X_train)
                # normalization
                for jj in range(p):
                    X_train[:, jj] = X_train[:, jj] - np.mean(X_train[:, jj])
                std = np.sqrt(np.sum(X_train * X_train, axis=0)) / np.sqrt(n)
                Xmat = X_train / std
                yval_mat = np.concatenate(
                    (
                        np.array([x for x in sample_i_aj["s2"]]),
                        np.array(sample_i_aj["r"]).reshape(-1, 1),
                    ),
                    1,
                )
                dimp = np.shape(Xmat)[1]

                ########################  Main Loop  ########################################
                original_stdout = sys.stdout
                sys.stdout = StringIO()   ### avoid lasso.train() print useless information in simulation

                lambda_list = np.exp(np.arange(-10, 5, 0.1))
                beta_mat_states = np.empty(shape=(p + 1, p + 1), dtype=np.float64)
                for ith_state in range(p + 1):
                    yval = yval_mat[:, ith_state]
                    if env_name in ["AR1", "Mixed"]:
                        lasso = pycasso.Solver(
                            Xmat, yval - np.mean(yval), penalty="l1", lambdas=lambda_list
                        )

                        lasso.train()
                        BIC = np.zeros(len(lambda_list))
                        for k in range(len(lambda_list)):
                            BIC[k] = np.sum(
                                np.square(
                                    yval - np.mean(yval) - Xmat @ lasso.coef()["beta"][k]
                                )
                            ) + sum(lasso.coef()["beta"][k] != 0) * np.log(n)
                        beta_mat_states[ith_state] = np.array([0] + lasso.coef()["beta"][np.argmin(BIC)].tolist()).reshape(1, -1)
                        if VERBOSE:
                            print("BIC index:", np.argmin(BIC))
                    if env_name in ['CartPole-v0', 'LunarLander-v2']:
                        rf = RandomForestRegressor(
                            n_estimators=int(n / 30),
                            max_depth=4,
                            max_features="sqrt",
                        )
                        rf.fit(Xmat, yval)
                        beta_mat_states[ith_state] = np.array(
                            [0] + (rf.feature_importances_).tolist()
                        ).reshape(1, -1)

                beta_mat[j] = np.max(np.abs(beta_mat_states), 0)

                sys.stdout = original_stdout
                ################################################################

            max_beta = np.max(np.abs(beta_mat), 0)[1:]
            rest_list = list(range(p))
            Wi = max_beta[rest_list]
            next_ith_Ghat = [
                rest_list[i] for i, x in enumerate((Wi > 2.5).tolist()) if x == True
            ]

            G_set_all.append(set(next_ith_Ghat))
            print("Final Lasso Set based on Split Sample {}: ".format(str(ith)), next_ith_Ghat)

        if METHOD == "SFS":
            ### SFS parameter ###
            GAMMA = 0.99

            num_act = len(action_space)

            train_state = np.array(sample_i["s1"].tolist())
            train_state = StandardScaler(copy=True).fit_transform(train_state)
            train_next_state = np.array(sample_i["s1"].tolist())
            train_next_state = StandardScaler(copy=True).fit_transform(train_next_state)
            train_reward = np.array(sample_i["r"].tolist())
            train_reward = (
                StandardScaler(copy=True)
                .fit_transform(train_reward.reshape(-1, 1))
                .flatten()
            )
            train_action = np.array(sample_i["a"].tolist())
            train_done = np.array(sample_i["done"].tolist())

            train_data = [
                [
                    train_state[train_action == ith_act, :],
                    train_reward[train_action == ith_act],
                    train_next_state[train_action == ith_act, :],
                    train_done[train_action == ith_act],
                ]
                for ith_act in action_space
            ]

            ## Initialize Q-function to be zero and construct target
            regrs = []
            for ith_act in range(num_act):
                regr_i = ElasticNet()
                regr_i.fit(train_data[ith_act][0], train_data[ith_act][1])
                regrs.append(regr_i)

            ## FQI
            p = train_state.shape[1]
            for _ in range(10):
                y = list()
                for ith_act in range(num_act):
                    y.append(
                        GAMMA * regrs[ith_act].predict(train_next_state).reshape(-1, 1)
                    )
                y = np.hstack(y)
                y = np.max(y, axis=1)
                y = train_reward + (1 - train_done) * y

                regrs = []
                for ith_act in action_space:
                    regr_i = ElasticNet()
                    regrs.append(
                        regr_i.fit(
                            train_state[train_action == ith_act, :],
                            y[train_action == ith_act,],
                        )
                    )

            beta_mat = np.empty(shape=(num_act, p), dtype=np.float64)
            for ith_act in range(num_act):
                beta_mat[ith_act] = regrs[ith_act].coef_

            max_beta = np.max(np.abs(beta_mat), 0)

            vote = [
                list(range(p))[i]
                for i, x in enumerate((max_beta > 0).tolist())
                if x == True
            ]

        if METHOD == "DRL":
            vote = list(range(p))

    tstop = datetime.now()
    speed = (tstop - tstart).seconds / 60

    # majority vote
    if METHOD not in ["SFS", "DRL"]:
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

    knockoff_list = vote

    ########## policy evaluation ##########
    def policy(obs, regr0, regr1):
        q0 = regr0.predict(np.expand_dims(obs, 0))
        q1 = regr1.predict(np.expand_dims(obs, 0))
        if q0 > q1:
            return 0
        else:
            return 1

    def reward(regr0, regr1, knockoff_list, episodes=1000, env_name="CartPole-v0"):
        '''
        online evaluation
        '''
        if env_name == "CartPole-v0" or env_name == "MountainCar-v0" or env_name == "Acrobot-v1":
            MAXIMUM_STEP = 200
            rewards = 0
            for i in range(episodes):
                obs = get_init_state(env, i*seed+i)
                ar_state = np.random.normal(
                    _mu, _sigma, p - len(env.observation_space.low)
                )
                done = False
                step = 0
                while not done and step < MAXIMUM_STEP:
                    obs_nk = np.concatenate((obs, ar_state))[knockoff_list]
                    a = policy(obs_nk, regr0, regr1)
                    result_step = env.step(a)
                    if len(result_step) == 4:
                        obs, r, done, info = result_step
                    else:
                        obs, r, done, info, _ = result_step
                    step += 1
                    ar_state = ar_cons * ar_state + np.random.normal(
                        _mu, _sigma, p - len(env.observation_space.low)
                    )

                    rewards += r
            rewards = rewards / episodes
            return rewards
        if env_name == "AR1":
            rewards = 0
            state = np.random.normal(_mu, _sigma, p)
            i = 0
            while i < episodes:
                obs_nk = state[knockoff_list]
                a = policy(obs_nk, regr0, regr1)
                r = a * (state[0] + state[1]) + np.random.normal(_mu, _sigma, 1)[0]
                if a == 1:
                    state = ar_cons * state + np.random.normal(_mu, _sigma, p)
                if a == 0:
                    state = ar_cons * state / 10 + np.random.normal(_mu, _sigma, p)

                rewards += r
                i = i + 1
            rewards = rewards / episodes
            return rewards
        if env_name == "Mixed":
            states = [1, 2]
            n_1_1_1 = 0.2
            n_1_1_2 = 0.8
            n_1_2_1 = 0.3
            n_1_2_2 = 0.7
            n_2_1_1 = 0.55
            n_2_1_2 = 0.45
            n_2_2_1 = 0.35
            n_2_2_2 = 0.65
            transition_ar = [
                [[n_1_1_1, n_1_1_2], [n_1_2_1, n_1_2_2]],
                [[n_2_1_1, n_2_1_2], [n_2_2_1, n_2_2_2]],
            ]
            rewards = 0
            initial_state = list(np.zeros(p))
            for i in range(p):
                initial_state[i] = np.random.choice(states, replace=True, p=[0.5, 0.5])
            current_state = initial_state
            obs_nk = deepcopy(current_state)
            ith_eps = 0
            while ith_eps < episodes:
                a = policy(np.array(obs_nk)[knockoff_list], regr0, regr1)
                r = a + current_state[0] + np.random.normal(loc=0, scale=0.5)

                p_by_state1 = np.exp(current_state[1]) / (np.exp(current_state[1]) + 1)

                current_state[0] = np.random.choice([1, 2], replace=True, p=[p_by_state1, 1-p_by_state1])
                current_state[1] = np.random.choice([1, 2], replace=True, p=[0.5, 0.5])

                for i in range(2, 2 + math.floor((p - 2) / 2)):
                    current_state[i] = np.random.choice(
                        [1, 2],
                        replace=True,
                        p=transition_ar[a - 1][current_state[i] - 1],
                    )

                for i in range(2 + math.floor((p - 2) / 2), p):
                    current_state[i] = np.random.binomial(1, 0.5, None) + 1

                obs_nk = deepcopy(
                    (
                        np.array(current_state)
                        + np.random.normal(0, 0.1, len(current_state))
                    ).tolist()
                )
                rewards += r
                ith_eps = ith_eps + 1
            rewards = rewards / episodes
            return rewards

    if POLICY_EVALUATION and len(knockoff_list) > 0:
        np.random.seed(seed)
        random.seed(seed)

        sar = []

        ### Offline data preparation
        if env_name == "CartPole-v0" or env_name == "MountainCar-v0" or env_name == "Acrobot-v1":
            MAXIMUM_STEP = 200
            env = gym.make(env_name)
            for i in range(1, n_traj + 1):

                state = get_init_state(env, i*seed+i)
                ar_state = np.random.normal(
                    _mu, _sigma, p - len(env.observation_space.low)
                )

                a = env.action_space.sample()
                step = 0
                while True:
                    result_step = env.step(a)
                    if len(result_step) == 4:
                        next_state, r, done, _ = result_step  # for conda_python3
                    else:
                        next_state, r, done, _, _ = result_step
                    step += 1

                    state1 = np.concatenate((state, ar_state))[knockoff_list]
                    ar_state = ar_cons * ar_state + np.random.normal(
                        _mu, _sigma, p - len(env.observation_space.low)
                    )
                    state2 = np.concatenate((next_state, ar_state))[knockoff_list]

                    if done or step >= MAXIMUM_STEP:
                        r = 0
                        sar.append([state1, a, r, done])

                        break

                    sar.append([state1, a, r, done])

                    next_a = opt_policy(next_state, epsilon=eps)[0]

                    a = next_a
                    state = next_state

        if env_name == "AR1":
            state = np.random.normal(_mu, _sigma, p)
            i = 0
            max_T = 150
            while i < n_traj:
                t = 0
                while t < max_T:

                    if (state[0] + state[1]) > 0:
                        a = np.random.binomial(1, 1 - eps, 1)[0]
                    else:
                        a = np.random.binomial(1, eps, 1)[0]

                    r = a * (state[0] + state[1]) + np.random.normal(_mu, _sigma, 1)[0]

                    sar.append([state[knockoff_list], a, r, False])
                    if a == 1:
                        state = ar_cons * state + np.random.normal(_mu, _sigma, p)
                    if a == 0:
                        state = ar_cons * state / 10 + np.random.normal(_mu, _sigma, p)
                    t = t + 1

                if (state[0] + state[1]) > 0:
                    a = np.random.binomial(1, 1 - eps, 1)[0]
                else:
                    a = np.random.binomial(1, eps, 1)[0]

                r = a * (state[0] + state[1]) + np.random.normal(_mu, _sigma, 1)[0]

                sar.append([state[knockoff_list], a, r, True])

                i = i + 1

        if env_name == "Mixed":
            states = [1, 2]
            actions = [1, 2]
            n_1_1_1 = 0.2
            n_1_1_2 = 0.8
            n_1_2_1 = 0.3
            n_1_2_2 = 0.7
            n_2_1_1 = 0.55
            n_2_1_2 = 0.45
            n_2_2_1 = 0.35
            n_2_2_2 = 0.65
            transition_ar = [
                [[n_1_1_1, n_1_1_2], [n_1_2_1, n_1_2_2]],
                [[n_2_1_1, n_2_1_2], [n_2_2_1, n_2_2_2]],
            ]
            ith_traj = 0
            max_T = 150
            while ith_traj < n_traj:
                t = 0
                initial_state = list(np.zeros(p))

                for i in range(p):
                    initial_state[i] = np.random.choice(
                        states, replace=True, p=[0.5, 0.5]
                    )

                current_state = initial_state

                next_state = deepcopy(current_state)

                while t < max_T:
                    a = np.random.choice(actions, replace=True, p=[0.5, 0.5])
                    r = a + current_state[0] + np.random.normal(loc=0, scale=0.5)
                    sar.append([np.array(next_state)[knockoff_list], a-1, r, False])
                    p_by_state1 = np.exp(current_state[1]) / (
                        np.exp(current_state[1]) + 1
                    )
                    current_state[0] = np.random.choice(
                        [1, 2], replace=True, p=[p_by_state1, 1-p_by_state1]
                    )
                    current_state[1] = np.random.choice(
                        [1, 2], replace=True, p=[0.5, 0.5]
                    )
                    for i in range(2, 2 + math.floor((p - 2) / 2)):
                        current_state[i] = np.random.choice(
                            [1, 2],
                            replace=True,
                            p=transition_ar[a - 1][current_state[i] - 1],
                        )

                    for i in range(2 + math.floor((p - 2) / 2), p):
                        current_state[i] = np.random.binomial(1, 0.5, None) + 1

                    next_state = deepcopy(
                        (
                            np.array(current_state)
                            + np.random.normal(0, 0.1, len(current_state))
                        ).tolist()
                    )

                    t = t + 1

                a = np.random.choice(actions, replace=True, p=[0.5, 0.5])
                r = a + current_state[0] + np.random.normal(loc=0, scale=0.5)
                sar.append([np.array(next_state)[knockoff_list], a-1, r, True])
                ith_traj = ith_traj + 1

        ### FQI
        GAMMA = 0.99
        num_act = 2

        ### data preparation
        train_state = np.array([x[0] for x in sar])
        train_state = StandardScaler(copy=True).fit_transform(train_state)
        train_next_state = np.roll(train_state, shift=-1, axis=0)
        train_reward = np.array([x[2] for x in sar])
        train_reward = (
            StandardScaler(copy=True)
            .fit_transform(train_reward.reshape(-1, 1))
            .flatten()
        )
        train_action = np.array([x[1] for x in sar])
        train_done = np.array([x[3] for x in sar])

        ## Initialize Q-function to be zero and construct target
        regrs = []
        for ith_act in range(num_act):
            regr_i = MLPRegressor(
                random_state=1,
                max_iter=50,
                warm_start=True,
                # early_stopping=True,
                # hidden_layer_sizes=(128,),
                # alpha=0.0003,
            )
            regr_i.fit(train_state[train_action==ith_act, :], 
                       train_reward[train_action==ith_act])
            regrs.append(regr_i)

        ## TD update (learning optimal policy):
        rewardlist = np.zeros(10)
        for k in range(10):
            ## compute TD target
            y = []
            for ith_act in range(num_act):
                y.append(
                    GAMMA * regrs[ith_act].predict(train_next_state).reshape(-1, 1)
                )
            y = np.hstack(y)
            y = np.max(y, axis=1)   # TD-max operator
            y = train_reward + (1 - train_done) * y

            for ith_act in range(num_act):
                regrs[ith_act].fit(train_state[train_action==ith_act, :], y[train_action==ith_act, ])

            rewardlist[k] = reward(
                regr0=regrs[0],
                regr1=regrs[1],
                episodes=eval_episode,
                knockoff_list=knockoff_list,
                env_name=env_name,
            )
            print("k:", k, "kth_value:", rewardlist[k])

    else:
        rewardlist = 0

    print(G_set_all)
    print("Seed:", seed, "Estimated Ghat: ", vote, "Time Spent (Minutes): ", speed)  # , 'Averaged Value: ', value)

    return G_set_all, vote, speed, rewardlist, num_splits


# %% Large-scale dataset
ENV_LIST = [
    "CartPole-v0",
    "LunarLander-v2",
    "AR1",
    "Mixed",
]
METHOD_LIST = ["SFS", "RewardOnly", "SEEK", "VS", "SFS", "DRL"]
N_P_LIST = [(600, 600), (800, 800), (1000, 1000)]
AR_CONS_LIST = [0.0, 0.9]
rep_number = 10    # we repeat 10 times as the results on large scale datasets are rather stable. 
for n_p in N_P_LIST:
    for ar_cons in AR_CONS_LIST:
        for env_name in ENV_LIST:
            for method in METHOD_LIST:
                n_traj, p = n_p
                print("Method: {} under {} with n={}; ar={} Begin!".format(method, env_name, str(n_traj), str(ar_cons)))
                settings = [p, n_traj, ar_cons, env_name, method]
                rep_res = []
                for seed_i in range(rep_number):
                    res_i = exp_knock(settings, seed_i)
                    rep_res.append(res_i)

                file_name = "res/LargeScale_{}_{}_N_{}_p_{}_ar_{}.pkl".format(method, env_name, str(n_traj), str(p), str(ar_cons))
                with open(file_name, "wb") as filehandle:
                    pickle.dump(rep_res, filehandle)
                print("Method: {} under {} with n={}; ar={} Done!".format(method, env_name, str(n_traj), str(ar_cons)))

# %% AR and Mixed Environments (low dimension)
env_list = ["AR1", "Mixed"]
for env_name in env_list:
    METHOD_LIST = ['RewardOnly', 'SEEK', 'OneStep', 'VS', 'SFS', 'DRL']
    if env_name == "AR1":
        N_TRAJ_LIST = [10, 20, 40]
    if env_name == "Mixed":
        N_TRAJ_LIST = [50, 100, 200]

    p = 20
    ar_cons = 0.9
    rep_number = 50

    for method in METHOD_LIST:
        for n_traj in N_TRAJ_LIST:
            settings = [p, n_traj, ar_cons, env_name, method]

            rep_res = []
            for seed_i in range(rep_number):
                res_i = exp_knock(settings, seed_i)
                rep_res.append(res_i)

            with open("res/{}_{}_whole_NumTraj{}p{}_1120.pkl".format(method, env_name, str(n_traj), str(p)), "wb") as filehandle:
                pickle.dump(rep_res, filehandle)

# %% CartPole and LunarLander Environments (with parallelism)
if __name__ == '__main__':

    # env_name = "LunarLander-v2"
    env_name = "CartPole-v0"

    N_TRAJ_LIST = [100, 200]
    P_LIST = [50, 100, 150, 200]
    METHOD_LIST = [
        "RewardOnly",
        "OneStep",
        "SEEK",
        "SFS",
        "DRL",
        "VSLASSO",
    ]

    if env_name == "CartPole-v0":
        AR_CONS_LIST = [0.0, 0.9]
    else:
        AR_CONS_LIST = [0.0, 0.9]

    rep_number = 20
    NUM_THREAD = 10

    for method in METHOD_LIST:
        for ar_cons in AR_CONS_LIST:
            for n_traj in N_TRAJ_LIST:
                for p in P_LIST:
                    settings = [p, n_traj, ar_cons, env_name, method]

                    # # multi-processing:
                    # with Pool(NUM_THREAD) as pool:
                    #     rep_res = list(tqdm(pool.imap(partial(exp_knock, settings), range(rep_number)), total=rep_number))

                    # single-processing:
                    rep_res = []
                    for seed_i in tqdm(range(rep_number)):
                        res_i = exp_knock(settings, seed_i)
                        rep_res.append(res_i)

                    date_time = datetime.fromtimestamp(datetime.timestamp(datetime.now())).strftime("%Y%m%d%H%M%S")

                    with open(
                        "res/{}_{}_whole_NumTraj_{}_p_{}_ar_{}_{}.pkl".format(
                            method, env_name, str(n_traj), str(p), str(ar_cons), date_time
                        ),
                        "wb",
                    ) as filehandle:
                        pickle.dump(rep_res, filehandle)
