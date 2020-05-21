#!/usr/bin/env python
# coding: utf-8

import random
import sys
from time import time
from collections import deque, defaultdict, namedtuple
import numpy as np
import pandas as pd
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# In[2]:


def set_env_seed(x):
    env = gym.make('LunarLander-v2')
    env.seed(x)
    return env


# In[4]:

def set_device():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print("RUNNING ON ",device)
    return device



#-------------Q VALUE APPROXIMATION NETWORK-------------------------------
class QNet(nn.Module):
    def __init__(self, state_size, action_size, seed,n_fc1,n_fc2,af1,af2):
        super(QNet, self).__init__()
        torch.manual_seed(seed)
        self.n_fc1 = n_fc1
        self.n_fc2 = n_fc2
        self.fc1 = nn.Linear(state_size, self.n_fc1)
        self.fc2 = nn.Linear(self.n_fc1, self.n_fc2)
        self.fc3 = nn.Linear(self.n_fc2, action_size)
        self.af1 = af1
        self.af2 = af2

    def forward(self, x):
        """Forward pass"""
        if self.af1==0:
            x = F.relu(self.fc1(x))
        else:
            x = torch.sigmoid(self.fc1(x))
        if self.af2==0:
            x = F.relu(self.fc2(x))
        else:
            x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)

        return x


# In[7]:

#-------------EXPERIENCE SAMPLER-------------------------------------------
class ExperienceReplays:
    def __init__(self, buffer_size, batch_size, seed,device):
        self.device = device
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):

        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):

        experiences = random.sample(self.memory, k=self.batch_size)

        # Convert to torch tensors
        states = torch.from_numpy(np.vstack([experience.state for experience in experiences if experience is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([experience.action for experience in experiences if experience is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([experience.reward for experience in experiences if experience is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([experience.next_state for experience in experiences if experience is not None])).float().to(self.device)
        # Convert done from boolean to int
        dones = torch.from_numpy(np.vstack([experience.done for experience in experiences if experience is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


# In[8]:

#-------------------------DUEL DQN ALGORITHM---------------------------------
class DDQN:
    def __init__(self, state_size, action_size, seed,n_fc1,n_fc2,af_1,af_2,device, BUFFER_SIZE, BATCH_SIZE,GAMMA, TAU, LR, UPDATE_EVERY):
        #self.learn_mode = learn_mode
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = UPDATE_EVERY
        self.TAU = TAU
        self.LR = LR
        #--------Initialize Q and Fixed Q networks with same architecture---------
        print("Device check - ",self.device)
        self.q_network = QNet(state_size, action_size, seed, n_fc1, n_fc2,af_1,af_2).to(self.device)
        print("Training....",self.q_network)
        self.fixed_network = QNet(state_size, action_size, seed, n_fc1, n_fc2,af_1,af_2).to(self.device)
        #self.flag_1 = 0
        #self.flag_2 = 0
        #print("------------Initializing if some warm start is present.......")
        try:
            fc1_weight = load_pickle(str(n_fc1)+"_"+str(af_1)+"fc1_weight.pkl")
            #print(fc1_weight)
            fc1_bias = load_pickle(str(n_fc1)+"_"+str(af_1)+"fc1_bias.pkl")
            #print("Shared weight fc1 history found {}_{}".format(n_fc1,af_1))
            #self.flag_1 = 1
            with torch.no_grad():
                self.q_network.fc1.weight.copy_(fc1_weight)
                self.q_network.fc1.bias.copy_(fc1_bias)
                #print("Copied to Q L1")
                #print(self.q_network.fc1.weight)
                    # self.q_network.fc2.weight.copy_(fc2_weight)
                    # self.q_network.fc2.bias.copy_(fc2_bias)
                self.fixed_network.fc1.weight.copy_(fc1_weight)
                self.fixed_network.fc1.bias.copy_(fc1_bias)
                #print("Copied to fixed Q L1")
                    # self.fixed_network.fc2.weight.copy_(fc2_weight)
                    # self.fixed_network.fc2.bias.copy_(fc2_bias)
        except:
            #print("No layer 1 shared history found {}_{}".format(n_fc1,af_1))
            pass
        try:
            fc2_weight = load_pickle(str(n_fc2)+"_"+str(af_2)+"fc2_weight.pkl")
            fc2_bias = load_pickle(str(n_fc2)+"_"+str(af_2)+"fc2_bias.pkl")
            #print("Shared weight fc2 history found {}_{}".format(n_fc2,af_2 ))
            #self.flag_2 = 1
            with torch.no_grad():
                # self.q_network.fc1.weight.copy_(fc1_weight)
                # self.q_network.fc1.bias.copy_(fc1_bias)
                self.q_network.fc2.weight.copy_(fc2_weight)
                self.q_network.fc2.bias.copy_(fc2_bias)
                #print("Copied to Q L2")
                # self.fixed_network.fc1.weight.copy_(fc1_weight)
                # self.fixed_network.fc1.bias.copy_(fc1_bias)
                self.fixed_network.fc2.weight.copy_(fc2_weight)
                self.fixed_network.fc2.bias.copy_(fc2_bias)
                #print("Copied to fiexed Q L2")
        except:
            #print("No layer 2 shared history found {}_{}".format(n_fc2,af_2))
            pass




        self.optimizer = optim.Adam(self.q_network.parameters(),lr=self.LR)
        self.memory = ExperienceReplays(self.BUFFER_SIZE, self.BATCH_SIZE, seed, self.device)  #alloting memory for experience buffer
        self.timestep = 0
        #return flag_1,flag_2


    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)   #adds experience to replay buffer
        self.timestep += 1
        if self.timestep % self.UPDATE_EVERY == 0:
            if len(self.memory) > self.BATCH_SIZE:
                sampled_experiences = self.memory.sample()   #randomly samples from experience
                self.learn(sampled_experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        action_values = self.fixed_network(next_states).detach()
        max_action_values = action_values.max(1)[0].unsqueeze(1)
        Q_target = rewards + (self.GAMMA * max_action_values * (1 - dones))
        Q_expected = self.q_network(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
         # Soft parameter update of fixed network
        self.update_fixed_network(self.q_network, self.fixed_network)


    def update_fixed_network(self, q_network, fixed_network):
        for source_parameters, target_parameters in zip(q_network.parameters(), fixed_network.parameters()):
            target_parameters.data.copy_(self.TAU * source_parameters.data + (1.0 - self.TAU) * target_parameters.data)


    def epsilor_greedy_act(self, state, eps=0.0):
         #-----epsilon greedy-------------------------------------
        rnd = random.random()
        if rnd < eps:
            return np.random.randint(self.action_size)
        else:

            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            #---set the network into evaluation mode
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            #----choose best action
            action = np.argmax(action_values.cpu().data.numpy())
            #----We need switch it back to training mode
            self.q_network.train()
            return action

    def checkpoint(self, filename):
        torch.save(self.q_network.state_dict(), filename)







#----------------------------------TRAINING-------------------------------------------
def train_dqn(n_fc1,n_fc2,af_1,af_2,env_seed):
    BUFFER_SIZE = int(1e5) # Replay memory size
    BATCH_SIZE = 64         # Number of experiences to sample from memory
    GAMMA = 0.99           # Discount factor
    TAU = 1e-3              # Soft update parameter for updating fixed q network instead of updating fixed  Q network after some steps
    LR = 1e-3               # Q Network learning rate
    UPDATE_EVERY = 5        # How often to update Q network
    MAX_EPISODES =  1700  # Max number of episodes to play
    MAX_STEPS = 900     # Max steps allowed in a single episode/play
    ENV_SOLVED = 200     # MAX score at which we consider environment to be solved
    PRINT_EVERY = 100    # How often to print the progress
    EPS_START = 1.0      # Default/starting value of eps
    EPS_DECAY = 0.999    # Epsilon decay rate - this decay rate is selected so as
    EPS_MIN = 0.01       # Minimum epsilon
    env = set_env_seed(env_seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = set_device()
    print("Device initialised - ", device)
    dqn_agent = DDQN(state_size, action_size, 0,n_fc1,n_fc2,af_1,af_2,device, BUFFER_SIZE, BATCH_SIZE,GAMMA, TAU, LR, UPDATE_EVERY)
    f1 = open("non_converging_models.txt",'r')
    ncm = f1.readlines()
    f1.close()
    if str(n_fc1)+"_"+str(n_fc2)+"_"+str(af_1)+"_"+str(af_2) in ncm:
        ncm_flag = 1
    else:
        ncm_flag = 0

    try:
        #dqn_agent1 = DDQN(state_size, action_size, 0,n_fc1,n_fc2,af_1,af_2,device)
        dqn_agent.q_network.load_state_dict(torch.load('solved_200_'+str(n_fc1)+'_'+str(n_fc2)+'_'+str(af_1)+"_"+str(af_2)+'.pth'))
        print("Model already trained!")
    except:
        if ncm_flag == 0:
            solved_flag = 0
            arch = str(n_fc1)+"_"+str(n_fc2)+"_"+str(af_1)+"_"+str(af_2)
            start = time()
            scores = []
            scores_window = deque(maxlen=100)
            eps = EPS_START
            for episode in tqdm(range(1, MAX_EPISODES + 1)):
                state = env.reset()
                score = 0
                for t in range(MAX_STEPS):
                    action = dqn_agent.epsilor_greedy_act(state, eps)
                    next_state, reward, done, info = env.step(action)
                    dqn_agent.step(state, action, reward, next_state, done)
                    state = next_state
                    score += reward
                    if done:
                        break

                    eps = max(eps * EPS_DECAY, EPS_MIN)
                    if episode % PRINT_EVERY == 0:
                        mean_score = np.mean(scores_window)
                        print('\r{} architecture Progress {}/{}, average score:{:.2f}'.format(arch,episode, MAX_EPISODES, mean_score), end="")
                    if score >= ENV_SOLVED:
                        solved_flag = 1
                        mean_score = np.mean(scores_window)
                        print('\r{} architecture Environment solved in {} episodes, average score: {:.2f}'.format(arch,episode, mean_score), end="")
                        sys.stdout.flush()
                        dqn_agent.checkpoint('solved_200_'+str(n_fc1)+'_'+str(n_fc2)+'_'+str(af_1)+"_"+str(af_2)+'.pth')
                        break

                scores_window.append(score)
                scores.append(score)
            end = time()
            print('Took {} seconds'.format(end - start))
            time_taken = end - start
            if solved_flag ==0:
                f=open("non_converging_models.txt","a")
                f.write(str(n_fc1)+"_"+str(n_fc2)+"_"+str(af_1)+"_"+str(af_2)+'\n')
                f.close()
            return time_taken
        else:
            print("non converging model")
            pass




# In[16]:


# plt.figure(figsize=(10,6))
# plt.plot(scores)
# # A bit hard to see the above plot, so lets smooth it (red)
# plt.plot(pd.Series(scores).rolling(100).mean())
# plt.title('DQN Training'+str(sys.argv[1])+'_'+str(sys.argv[2]))
# plt.xlabel('# of episodes')
# plt.ylabel('score')
# plt.savefig('solved_200_'+str(sys.argv[1])+'_'+str(sys.argv[2])+'_'+str(sys.argv[3])+'.png')


# In[28]:





# In[31]:
#------------------PLAYING GAME VIDEO CODE---------------------------------------------------------

# from gym import wrappers
# if env:

#     env.close()
#     env = gym.make('LunarLander-v2')
#     env.seed(0)
#     env = wrappers.Monitor(env, '/tmp/lunar-lander-6', video_callable=lambda episode_id: True,force=True)


# In[30]:
import pickle

def load_pickle(pkl_name):
    readfile = open(pkl_name, 'rb')
    model = pickle.load(readfile)
    return model

def dump_pickle(obj,name):
    file = open(name+str('.pkl'), 'wb')
    pickle.dump(obj, file)

#--------------TESTING------------------------------------------------------------------------------
def test(n_fc1,n_fc2,af_1,af_2,mm):
    BUFFER_SIZE = int(1e5) # Replay memory size
    BATCH_SIZE = 64         # Number of experiences to sample from memory
    GAMMA = 0.99           # Discount factor
    TAU = 1e-3              # Soft update parameter for updating fixed q network instead of updating fixed  Q network after some steps
    LR = 9e-4               # Q Network learning rate
    UPDATE_EVERY = 5 
    try:
        device = set_device()
        
        dqn_agent = DDQN(8,4, 0,n_fc1,n_fc2,af_1,af_2,device,BUFFER_SIZE, BATCH_SIZE,GAMMA, TAU, LR, UPDATE_EVERY)
        dqn_agent.q_network.load_state_dict(torch.load('solved_200_'+str(n_fc1)+'_'+str(n_fc2)+'_'+str(af_1)+"_"+str(af_2)+'.pth'))
        #dqn_agent.eval()
        print("model_loaded")
        dump_pickle(dqn_agent.q_network.state_dict()['fc1.weight'],str(n_fc1)+"_"+str(af_1)+"fc1_weight")
        dump_pickle(dqn_agent.q_network.state_dict()['fc1.bias'],str(n_fc1)+"_"+str(af_1)+"fc1_bias")
        dump_pickle(dqn_agent.q_network.state_dict()['fc2.weight'],str(n_fc2)+"_"+str(af_2)+"fc2_weight")
        dump_pickle(dqn_agent.q_network.state_dict()['fc2.bias'],str(n_fc2)+"_"+str(af_2)+"fc2_bias")
        print("Shared weights dumped for ",str(n_fc1)+"_"+str(af_1)+"_"+str(n_fc2)+"_"+str(af_2))
        #print('solved_200_'+str(n_fc1)+'_'+str(n_fc2)+'_'+str(af_1)+"_"+str(af_2)+'.pth')
        print("Calculating reward......")
        env = set_env_seed(799)
        total_score = 0
        try:
            r_dict = load_pickle("reward_dict.pkl")
        except:
            r_dict={}
            dump_pickle(r_dict,"reward_dict")
<<<<<<< HEAD

=======
>>>>>>> 781b26fab74e392d80e270f83302b73373f43abc
        search_key = str(n_fc1)+'_'+str(n_fc2)+'_'+str(af_1)+"_"+str(af_2)
        if  search_key in r_dict.keys():
            avg_score = r_dict[search_key]


        else:
            for i in range(500):
                score = 0
                state = env.reset()
                while True:

                    action = dqn_agent.epsilor_greedy_act(state)
                    next_state, reward, done, info = env.step(action)
                    state = next_state
                    score += reward
                    if done:
                        break
                total_score+=score
                    #print('episode: {} scored {}'.format(i, score))
            avg_score = total_score/500
            print("Average_Score = ",avg_score)
            f = open("logger_results_enas.txt",'a')
            f.write('solved_200_'+str(n_fc1)+'_'+str(n_fc2)+'_'+str(af_1)+"_"+str(af_2)+'\t'+str(total_score/500)+'\t'+'\n')

            #r_dict = load_pickle("reward_dict.pkl")
            search_key = str(n_fc1)+'_'+str(n_fc2)+'_'+str(af_1)+"_"+str(af_2)
            r_dict[search_key] = avg_score
            dump_pickle(r_dict,"reward_dict")



        score_component = abs((avg_score)/30)
        #time_component = (time/3500)
        return score_component
    except:
        print("--Didn't converge--")
        return 0.00001
# In[ ]:
