import torch
import gym 
import numpy as np
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression 

def get_init_state(env, seed):
    try:
        state, _ = env.reset(return_info=True)
    except TypeError:
        state = env.reset(seed=seed)
        if isinstance(state, tuple):
            state = state[0]
    return state

def save(args, save_name, model, wandb, ep=None):
    import os
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
        wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
        wandb.save(save_dir + args.run_name + save_name + ".pth")


def collect_random(env, dataset, episodes, knockoff_list, opt_policy, eps = 0.1, ar_cons = 0, _mu=0, _sigma=0.5, p=10):
    
    n_actions = env.action_space.n
    actions = np.arange(n_actions) 
    dim = len(knockoff_list)
    
    for i in range(1, episodes + 1):

        state = env.reset()
        ar_state = np.random.normal(_mu, _sigma, p - len(env.observation_space.low))
        
        a = env.action_space.sample() 
        
        while True: 
            next_state, r, done, _ = env.step(a)  # check the openAI github repo
            
            state1 = np.concatenate((state, ar_state))[knockoff_list] 
            ar_state = ar_cons * ar_state + np.random.normal(_mu, _sigma, p - len(env.observation_space.low))
            state2 = np.concatenate((next_state, ar_state))[knockoff_list] 

            if done: 

                r=0  
                dataset.add(state1, a, r, state2, done)
                 
                break 

            dataset.add(state1, a, r, state2, done)      
 
            next_a = opt_policy(next_state, epsilon=eps)[0] 

            a = next_a
            state = next_state              
            