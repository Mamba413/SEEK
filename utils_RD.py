import numpy as np
import pandas as pd
from d3rlpy.dataset import MDPDataset

# %%
def create_Ohio_dataset(var_indices, action_continuous=False):
    list_patients = list(range(6))
    state1, acts, rwds, dones = [], [], [], []
    for i_set in list_patients:
        data = pd.read_csv("ohioT1DM/patient" + str(i_set + 1) + ".csv")
        lag = 6
        ar_cons = 0.5
        _mu = 0
        _sigma = 1
        p_ar = 10
        p = 2 + 2 * (lag - 1) + p_ar

        max_T = 48
        ar_state = np.random.normal(_mu, _sigma, p_ar)

        for i in range(len(data)):

            if i < lag:
                continue

            if i % max_T < lag:
                continue

            if (i + 1) % max_T == 0:
                continue

            current_state = []
            next_state = []

            if i_set > 5:
                for j in range(lag - 1):
                    current_state = (
                        current_state
                        + (
                            data[["glucose", "acceleration"]].loc[i - (lag - 1 - j)]
                        ).tolist()
                    )
                current_state = (
                    current_state + (data[["glucose", "acceleration"]].loc[i]).tolist()
                )
            else:
                for j in range(lag - 1):
                    current_state = (
                        current_state
                        + (data[["glucose", "heart"]].loc[i - (lag - 1 - j)]).tolist()
                    )
                current_state = (
                    current_state + (data[["glucose", "heart"]].loc[i]).tolist()
                )
            current_state = current_state + ar_state.tolist()
            ar_state = ar_cons * ar_state + np.random.normal(_mu, _sigma, p_ar)
            current_state = np.array(current_state)
            state1.append(current_state[var_indices])

            act = data["bolus"].loc[i]
            acts.append(0 * (act <= 0) + 1 * (act > 0))
            rwds.append(data["icg"].loc[i])

            if (i + 1) % max_T == max_T - 1:
                dones.append(True)
            else:
                dones.append(False)

    action = np.array(acts)
    if action_continuous:
        action = action + np.random.normal(size=np.size(action), scale=1e-3)
    action = action.reshape(-1, 1)

    dataset = MDPDataset(
        observations=np.array(state1),
        actions=action,
        rewards=np.array(rwds),
        terminals=np.array(dones),
    )
    return dataset


def create_MIMIC3_dataset(var_indices, include_noise_var, action_continuous=False):
    if include_noise_var:
        FILE_PATH = 'MIMIC3-Result/preprocess_mimic3_manual.pkl'
    else:
        FILE_PATH = 'MIMIC3-Result/preprocess_mimic3.pkl'
    import pandas as pd
    dict_data = pd.read_pickle(FILE_PATH)
    state = np.array([dict_data["s1"][i] for i in range(len(dict_data))])
    action = np.array([dict_data["a"][i] for i in range(len(dict_data))])
    if action_continuous:
        action = action + np.random.normal(size=np.size(action), scale=1e-3)
    action = action.reshape(-1, 1)
    reward = np.array([dict_data["r"][i] for i in range(len(dict_data))])
    done = np.array([dict_data["done"][i] for i in range(len(dict_data))])
    if len(var_indices) > 1:
        selected_state = state[:, var_indices]
    else:
        selected_state = np.ones((len(dict_data), 1))
    dataset = MDPDataset(
        observations=selected_state,
        actions=action,
        rewards=reward,
        terminals=done,
    ) 
    return dataset

def generate_ar1(T, p_ar = 10):
    ar_cons=0.5
    _mu = 0
    _sigma = 1
    ar_state_seq = []
    ar_state = np.random.normal(_mu, _sigma, p_ar)
    ar_state_seq.append(ar_state)
    for i in range(T - 1):
        ar_state = ar_cons * ar_state + np.random.normal(_mu, _sigma, p_ar)
        ar_state_seq.append(ar_state)
    return np.vstack(ar_state_seq)