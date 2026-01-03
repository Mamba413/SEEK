import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
import scipy


def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)


def Beta_Mixing_Est_Gaussian(
    dataset, a, sam_size=10000, n_components=5, BIC_select=False
):
    """
    Estimate Beta-Mixing by Mix Gaussian Method
    @para:
    dataset: input data
    a: blocks of interest
    sam_size: sampling size in uniform distribution to calculate integral
    n_components: the number of component for estimating the density function
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

    ## Fit Gaussian Density
    if BIC_select and n_components > 1:
        param_grid = {
            "n_components": range(1, n_components + 1, 2),
        }
        grid_search_single = GridSearchCV(
            GaussianMixture(covariance_type='full', random_state=0), param_grid=param_grid, scoring=gmm_bic_score, refit=True, n_jobs=-1,
        )
        grid_search_joint = GridSearchCV(
            GaussianMixture(covariance_type='full', random_state=0), param_grid=param_grid, scoring=gmm_bic_score, refit=True, n_jobs=-1,
        )
        grid_search_single.fit(sample_single)
        den_single = grid_search_single.best_estimator_
        best_n_components_single = grid_search_single.best_params_['n_components']
        grid_search_joint.fit(sample_joint)
        den_joint = grid_search_joint.best_estimator_
        best_n_components_joint = grid_search_joint.best_params_['n_components']
    else:
        den_single = GaussianMixture(
            n_components=n_components, random_state=0, covariance_type="full"
        ).fit(sample_single)
        den_joint = GaussianMixture(
            n_components=n_components, random_state=0, covariance_type="full"
        ).fit(sample_joint)
        best_n_components_single = n_components
        best_n_components_joint = n_components

    # Sampling State from Reference Gaussian Distribution
    smp_S_1 = den_joint.sample(sam_size)[0]
    # np.random.multivariate_normal(den_joint.means_[0], sigma_1, sam_size)
    smp_S_2 = np.concatenate(
        (den_single.sample(sam_size)[0], den_single.sample(sam_size)[0]), 1
    )
    # np.random.multivariate_normal(den_joint.means_[0], sigma_2, sam_size)
    smp_S = np.concatenate((smp_S_1, smp_S_2), 0)

    p = dataset_s.shape[1]
    density_joint = 0
    for ith_mix in range(best_n_components_joint):
        ## Get Covariance Matrix
        sigma_1 = den_joint.covariances_[ith_mix]
        # Get pdf
        Gaus_pdf_joint = multivariate_normal(
            mean=den_joint.means_[ith_mix], cov=sigma_1, allow_singular=True
        )
        density_joint = (
            density_joint + Gaus_pdf_joint.pdf(smp_S) * den_joint.weights_[ith_mix]
        )
    
    density_single_1 = 0
    density_single_2 = 0
    for ith_mix in range(best_n_components_single):
        sigma_2 = den_single.covariances_[ith_mix]
        Gaus_pdf_single = multivariate_normal(
            mean=den_single.means_[ith_mix], cov=sigma_2, allow_singular=True
        )

        density_single_1 = (
            density_single_1
            + Gaus_pdf_single.pdf(smp_S[:, :(p)]) * den_single.weights_[ith_mix]
        )
        density_single_2 = (
            density_single_2
            + Gaus_pdf_single.pdf(smp_S[:, (p):]) * den_single.weights_[ith_mix]
        )

    beta_bnd = np.mean(
        np.abs(
            (density_joint - density_single_1 * density_single_2)
            / (density_joint + density_single_1 * density_single_2)
        )
    )

    epsilon = len(dataset[0]) / a * beta_bnd

    return beta_bnd, epsilon


def Beta_Mixing_Debiasing(est_betas, K_range, parametric_form="exp"):
    """
    Step 2: Beta Mixing Debiasing Procedure
    """
    c_0 = np.min(np.array(est_betas))
    if parametric_form == "exp":
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
    elif parametric_form == "poly":

        def fun_obj(paras):
            return np.mean(
                (
                    np.array(est_betas)
                    - paras[0]
                    - paras[1] * np.power(paras[2], -np.array(K_range))
                )
                ** 2
            )

        est_para = scipy.optimize.minimize(fun_obj, x0=[c_0, 1.0, 2.0]).x

    fitted_a = est_para[1]
    fitted_b = est_para[2]
    return fitted_a, fitted_b


def Select_Best_K(num, fitted_a, fitted_b, parametric_form, error_threshold):
    """
    Step 3: Select the best K according to Type-I error
    """
    if parametric_form == "exp":

        def adjust_fun(K_flag):
            return fitted_a * np.exp(fitted_b * K_flag)

    elif parametric_form == "poly":

        def adjust_fun(K_flag):
            return fitted_a * np.power(fitted_b, -K_flag)

    K_flag = 1
    error = adjust_fun(K_flag) * num / K_flag
    while error > error_threshold:
        K_flag = K_flag + 1
        error = adjust_fun(K_flag) * num / K_flag
    print("best K =", K_flag)
    return K_flag


def determine_best_K(
    data,
    K_max=15,
    n_components=1,
    BIC_select=False,
    parametric_form="exp",
    error_threshold=0.01,
):
    num = data[0].shape[0]
    K_range = range(1, K_max + 1)

    est_betas, est_eps = [], []
    for ith_K in K_range:
        ith_beta, ith_eps = Beta_Mixing_Est_Gaussian(
            data,
            ith_K,
            sam_size=10000,
            n_components=n_components,
            BIC_select=BIC_select,
        )
        est_betas.append(ith_beta)
        est_eps.append(ith_eps)

    fitted_a, fitted_b = Beta_Mixing_Debiasing(est_betas, K_range, parametric_form)

    best_K = Select_Best_K(
        num, fitted_a, fitted_b, parametric_form, error_threshold=error_threshold
    )
    return best_K
