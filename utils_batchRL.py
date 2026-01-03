"""
Utilitiy functions for RL algorithm
"""
from sklearn.neural_network import MLPRegressor 
from sklearn.linear_model import LogisticRegression
import numpy as np 
import pandas as pd

class BatchRL_env(object):

    def __init__(self, environment, mlp_max_iter=100, envir_type='real'):
        """ 
        :param mlp_max_iter: maximum number of iteration in multilayer perceptrons.
        :param envir_type: the environment, choosen from 'simu' for simulated data and 'real' for calibrated real data.
        """ 
        self.environment = environment
        self.envir_type = envir_type  
        self.mlp_max_iter = mlp_max_iter
         
        self.state1_observe = []
        self.action_taken = []
        self.reward_received = []
        self.state2_observe = [] 
        
        return
    
                    
    def initialize_training(self, train_data):
        """
        Initialize training process
        """    
        self.Cost_Dict = {}
        self.Q_nn = {}
        self.train_data = train_data
        return
        
    def get_cost(self, l, r, seed=1):
        """
        Collect the cost function.
        :return: the cost
        """
        if self.Cost_Dict.get(str(l) + ':' + str(r)) == None:
            if l == r:
                self.Cost_Dict[str(l) + ':' + str(r)] = 0
                
            else: 
                subdata = self.train_data[(self.train_data['at'] >= l / self.m) & (self.train_data['at'] <= r / self.m)]
                if len(subdata) == 0:
                    self.Cost_Dict[str(l) + ':' + str(r)] = 0
                else:
                    regr = MLPRegressor(hidden_layer_sizes=(10,10), random_state=seed, max_iter=self.mlp_max_iter).fit(np.array([x for x in subdata['xt']]), subdata['yt'])
                    y_fit = regr.predict(np.array([x for x in subdata['xt']]))
                    self.Cost_Dict[str(l) + ':' + str(r)] = sum((y_fit - subdata['yt']) ** 2)
                    self.Q_nn[str(l) + ':' + str(r)] = regr
           
        return self.Cost_Dict.get(str(l) + ':' + str(r))
    
    
    def get_prop_score(self, l, r, seed=1, act_method='logistic'):
        """
        Calculate the propensity score function for each interval.
        :return: the propensity score function
        """
        self.train_data[str(l) + ':' + str(r)] = 1 * ((self.train_data['at'] >= l / self.m) & (self.train_data['at'] <= r / self.m))
        regr = MLPRegressor(hidden_layer_sizes=(10,10), random_state=seed, max_iter= self.mlp_max_iter, activation=act_method).fit(np.array([x for x in self.train_data['xt']]), self.train_data[str(l) + ':' + str(r)])
        
        return regr
     
    
    def least_square_loss(self, tau, test_data, seed=1):
        """
        Use the left k-fold to calculate the least square loss function.
        :return: the Estimated Value
        """    
        self.test_data = test_data
        ls_loss = 0
        for i in range(len(tau)):
            l = tau[i] 
            r = tau[i + 1] if i < len(tau) - 1 else self.m

            subdata = self.test_data[(self.test_data['at'] >= l / self.m) & 
                                    (self.test_data['at'] < r / self.m)] if i < len(tau) - 1 else self.test_data[(self.test_data['at'] >= l / self.m) & (self.test_data['at'] <= r / self.m)]
            
            if len(subdata) > 0:
                    
                fitted_Q = self.Q_nn[str(l) + ':' + str(r)].predict(np.array([x for x in subdata['xt']]))   
                ls_loss += sum((subdata['yt'] - fitted_Q) ** 2) 
        return ls_loss 
    
    
    def evaluate(self, tau, test_data, seed=1):
        """
        Value Evaluation
        :return: the Estimated Value
        """    
        self.test_data = test_data
        V_hat = 0
        for i in range(len(tau)):
            l = tau[i] 
            r = tau[i + 1] if i < len(tau) - 1 else self.m
            #print('Processing interval: (', l / self.m, ',', r / self.m, ')...')

            subdata = self.test_data[(self.test_data['at'] >= l / self.m) & 
                                    (self.test_data['at'] < r / self.m)] if i < len(tau) - 1 else self.test_data[(self.test_data['at'] >= l / self.m) & (self.test_data['at'] <= r / self.m)]
                
            if len(subdata) > 0:
                prop_score = self.get_prop_score(l, r)
                
                if prop_score == 1:
                    prob_fit = 1
                else:
                    prob_fit = prop_score.predict(np.array([x for x in subdata['xt']]))   
                prob_fit = np.minimum(np.maximum(prob_fit, len(subdata['yt']) / len(self.test_data['yt'])), 1.)
                #print('fitted behavior prob: ', prob_fit)
            
                pi_star_ind = np.array([(self.policy_evaluate(x) >= l / self.m) * (self.policy_evaluate(x) < r / self.m) for x in subdata['xt']]) if i < len(tau) - 1 else np.array([(self.policy_evaluate(x) >= l / self.m) * (self.policy_evaluate(x) <= r / self.m) for x in subdata['xt']])
                    
                fitted_Q = self.Q_nn[str(l) + ':' + str(r)].predict(np.array([x for x in subdata['xt']]))  
                #print('fitted Q: ', fitted_Q)
                V_hat += sum(pi_star_ind / prob_fit * (subdata['yt'] - fitted_Q) + fitted_Q)
                #print('diff: ', (subdata['yt'] - fitted_Q))
        
        V_hat = V_hat / len(self.test_data['at'])
        #print('Estimated Value: ', V_hat)
        
        return V_hat

 
    def sample(self):
        """
        sample one traj from environment based on behavior policy.
        :return:
        """
        if self.envir_type == 'simu':
            context = self.environment.get_context()
            action = self.policy_behavior(context)
            reward = self.environment.get_reward(context, action)
            
        elif self.envir_type == 'real':
        
            state1, action, reward, state2 = self.environment.get_onesample()

            self.state1_observe.append(state1)
            self.action_taken.append(action)
            self.reward_received.append(reward)
            self.state2_observe.append(state2) 
        
    def get_dataset(self, n):
        """
        generate data from environment based on behavior policy.
        :return: the dataset
        """
        for i in range(n):
            self.sample()
            
        dataset = pd.DataFrame(columns=['s1', 'a', 'r', 's2'])
        dataset['s1'] = self.state1_observe
        dataset['a'] = self.action_taken
        dataset['r'] = self.reward_received
        dataset['s2'] = self.state2_observe
        return dataset


 