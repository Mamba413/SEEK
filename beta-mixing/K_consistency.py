import numpy as np
import scipy
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))

from argparse import ArgumentParser

def Beta_Mixing_Debiasing(est_betas, K_range, parametric_form="exp"):
    """
    Step 2: Beta Mixing Debiasing Procedure
    """
    c_0 = np.min(np.array(est_betas))
    if parametric_form == "exp":
        fitted_beta_para = np.polyfit(
            np.array(K_range),
            np.log(np.array(est_betas) - c_0 + 1e-10),
            1,
            w=np.array(est_betas) ** 2,
        )
        a_0 = fitted_beta_para[1]
        b_0 = fitted_beta_para[0]

        def fun_obj(paras):
            return np.mean(
                (np.array(est_betas) - paras[0] - paras[1] * np.exp(paras[2] * np.array(K_range)))
                ** 2
            )

        bnds = ((None, None), (0, None), (None, 0))
        est_para = scipy.optimize.minimize(fun_obj, x0=[c_0, a_0, b_0], bounds=bnds).x
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
    # print("best K =", K_flag)
    return K_flag

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--est-beta-dir', type=str, default='est_beta.txt')
    parser.add_argument('--NT', type=int, default=1e7)
    parser.add_argument('--delta', type=float, default=0.05)

    args = parser.parse_args()

    est_beta = np.loadtxt(args.est_beta_dir)
    est_beta = est_beta.flatten()

    fitted_a, fitted_b = Beta_Mixing_Debiasing(est_beta, list(range(1, est_beta.shape[0]+1)))
    best_K = Select_Best_K(
        args.NT, fitted_a, fitted_b, 'exp', error_threshold=args.delta
    )
    np.savetxt('select_K.txt', np.array([best_K], dtype=np.int8))


