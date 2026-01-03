from abc import ABC
import numpy as np
import pandas as pd
import pickle
import random
import gym
import copy
import math
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression 
from utils import get_init_state
 
class DataGenerator(ABC):
    def __init__(self, mu=0, sigma=0.5, p=10):
        self._mu = mu
        self._sigma = sigma        
        self.p = p
        self.state1_observe = []
        self.action_taken = []
        self.reward_received = []
        self.state2_observe = [] 
        self.done = [] 
        self.probabilities=0  
        
class DataGenerator_gym(DataGenerator):
    
    def __init__(self, mu=0, sigma=0.5, p=10):
        super(DataGenerator_gym, self).__init__(mu, sigma, p)  

    def get_samples_egreedy(self, seed, episodes, opt_policy, env_name='MountainCar-v0', continuous_reward=False, eps=0.1, ar_cons=0):
        MAXIMUM_STEP = 500
        if env_name == "Acrobot-v1":
            MAXIMUM_STEP = 100
        
        # Import and initialize Environment
        env = gym.make(env_name) 
        n_actions = env.action_space.n
        actions = np.arange(n_actions)
#         env.seed(seed)
        env.action_space.seed(seed)             
        dim = self.p  
         
        for i in range(1, episodes + 1):
            step = 0
            state = get_init_state(env, seed*i+i)
            ar_state = np.random.normal(self._mu, self._sigma, self.p - len(env.observation_space.low)) 
            
            # a = env.action_space.sample() 
            a = opt_policy(state, epsilon=eps)[0]
            
            while True: 
                result_step = env.step(a)
                if len(result_step) == 4: 
                    next_state, r, done, _ = result_step  # for conda_python3
                else:
                    next_state, r, done, _, _ = result_step
                
                if done or step >= MAXIMUM_STEP: 
                    # r=0 
                    self.state1_observe.append(np.concatenate((state, ar_state)))
                    self.action_taken.append(a)
                    self.done.append(done)
                    if continuous_reward == True:
                        self.reward_received.append(next_state[1])#(reward)
                    else:
                        self.reward_received.append(r)
                        
                    ar_state = ar_cons * ar_state + np.random.normal(self._mu, self._sigma, self.p - len(env.observation_space.low))     
                    self.state2_observe.append(np.concatenate((next_state, ar_state))) 
                    
                    break
                      
                step += 1
                state1 = np.concatenate((state, ar_state))
                self.state1_observe.append(state1)
                self.action_taken.append(a)
                self.done.append(done)
                if continuous_reward == True:
                    self.reward_received.append(next_state[1])
                else:
                    self.reward_received.append(r)
                
                ar_state = ar_cons * ar_state + np.random.normal(self._mu, self._sigma, self.p - len(env.observation_space.low))    
                state2 = np.concatenate((next_state, ar_state))
                self.state2_observe.append(state2) 

                next_a = opt_policy(next_state, epsilon=eps)[0]

                a = next_a
                state = next_state         
         
        
    def get_dataset(self, seed, n, opt_policy, env_name = 'MountainCar-v0', continuous_reward = False, eps = 0.1, ar_cons = 0):
         
        """
        generate data from environment based on behavior policy.
        :return: the dataset
        """
#         for i in range(n):
#             self.sample()
        
        self.get_samples_egreedy(seed=seed, episodes=n, opt_policy=opt_policy, env_name=env_name, continuous_reward=continuous_reward, eps=eps, ar_cons=ar_cons)
        
        dataset = pd.DataFrame(columns=['s1', 'a', 'r', 's2', 'done'])
        dataset['s1'] = self.state1_observe
        dataset['a'] = self.action_taken
        dataset['r'] = self.reward_received
        dataset['s2'] = self.state2_observe
        dataset['done'] = self.done
        return dataset

class DataGenerator_AR1(DataGenerator):
    
    def __init__(self, mu=0, sigma=0.5, p=10):
        super(DataGenerator_AR1, self).__init__(mu, sigma, p)  

    def get_samples_egreedy(self, seed, episodes, max_T=150, eps = 0.1, ar_cons = 0):
        
        # Import and initialize Environment 
        n_actions = 2
        actions = np.arange(n_actions) 
        dim = self.p  
        
        
        i = 0
        
        while i < episodes:
            
            state = np.random.normal(self._mu, self._sigma, self.p)
            
            t = 0
            
            while t < max_T:
            
                self.state1_observe.append(state)

                if (state[0] + state[1]) > 0:
                    a = np.random.binomial(1, 1 - eps, 1)[0]
                else:
                    a = np.random.binomial(1, eps, 1)[0]

                self.action_taken.append(a)

                r = a * (state[0] + state[1]) + np.random.normal(self._mu, self._sigma, 1)[0]

                if a == 1:
                    next_state = ar_cons * state + np.random.normal(self._mu, self._sigma, self.p)
                if a == 0:
                    next_state = ar_cons * state / 10 + np.random.normal(self._mu, self._sigma, self.p)    

                self.reward_received.append(r)

                self.state2_observe.append(next_state)
                
                if t == max_T -1:
                    self.done.append(True)
                else:
                    self.done.append(False)
                
                state = next_state 
                
                t = t + 1
            
            i = i + 1
            

    def get_dataset(self, seed, n, eps = 0.1, ar_cons = 0):
         
        """
        generate data from environment based on behavior policy.
        :return: the dataset
        """ 
        
        self.get_samples_egreedy(seed=seed, episodes=n, eps=eps, ar_cons=ar_cons)
        
        dataset = pd.DataFrame(columns=['s1', 'a', 'r', 's2', 'done'])
        dataset['s1'] = self.state1_observe
        dataset['a'] = self.action_taken
        dataset['r'] = self.reward_received
        dataset['s2'] = self.state2_observe
        dataset['done'] = self.done
        
        return dataset    
    
    
class DataGenerator_Toy(DataGenerator):
    
    def __init__(self, mu=0, sigma=0.5, p=10):
        super(DataGenerator_Toy, self).__init__(mu, sigma, p)  

    def get_samples_egreedy(self, seed, episodes, max_T=150, eps=0.1, ar_cons=0):
        np.random.seed(seed)
        # Import and initialize Environment 
        n_actions = 2
        actions = np.arange(n_actions) 
        dimension = int(self.p)
        total_rounds = int(max_T)
        
        rng = np.random.default_rng()

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
        
        ith_eps = 0 
        
        while ith_eps < episodes:  
            initial_state = list(np.zeros(dimension))
            for i in range(dimension):
                initial_state[i] = np.random.choice(states, replace = True, p = [0.5, 0.5])
                
            current_state = initial_state  
            
            state_trajectory = [copy.copy(current_state)]
            
            rounds_num = 0
            
            while rounds_num < total_rounds: 
                action = np.random.choice(actions, replace = True, p=[0.5, 0.5])
                self.action_taken.append(action)
                reward = action + current_state[0] + np.random.normal(loc=0, scale=0.5)
                self.reward_received.append(reward)

                p_by_state1 = np.exp(current_state[1])/(np.exp(current_state[1])+1)
                current_state[0] = np.random.choice([1, 2], replace=True, p=[p_by_state1, 1.0-p_by_state1])
                current_state[1] = np.random.choice([1, 2], replace=True, p=[0.5, 0.5])

                # AR structure:
                for i in range(2, 2 + math.floor((dimension-2)/2)):
                    current_state[i] = np.random.choice([1, 2], replace = True, p = transition_ar[action - 1][current_state[i]-1])

                # i.i.d. structure:
                for i in range (2 + math.floor((dimension-2)/2), dimension):
                    current_state[i] = rng.binomial(1, 0.5, None) + 1
                    
                state_trajectory.append(copy.deepcopy((np.array(current_state) + np.random.normal(0, 0.1, len(current_state))).tolist())) 
 
                if rounds_num == total_rounds -1:
                    self.done.append(True)
                else:
                    self.done.append(False) 
                
                rounds_num += 1
            
            ith_eps = ith_eps + 1    
            state_trajectory_1 = copy.deepcopy(state_trajectory)
            state_trajectory_1.pop(total_rounds)
            state_trajectory_2 = copy.deepcopy(state_trajectory)
            state_trajectory_2.pop(0)
            
            self.state1_observe.extend(state_trajectory_1) 
            self.state2_observe.extend(state_trajectory_2)      

    def get_dataset(self, seed, n, eps = 0.1, ar_cons = 0):
         
        """
        generate data from environment based on behavior policy.
        :return: the dataset
        """ 
        
        self.get_samples_egreedy(seed=seed, episodes=n, eps=eps, ar_cons=ar_cons)
        
        dataset = pd.DataFrame(columns=['s1', 'a', 'r', 's2', 'done'])
        dataset['s1'] = self.state1_observe
        dataset['a'] = self.action_taken
        dataset['r'] = self.reward_received
        dataset['s2'] = self.state2_observe
        dataset['done'] = self.done
        
        return dataset    
    