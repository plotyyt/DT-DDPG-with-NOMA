import multiprocessing
import threading
import tensorflow as tf
import numpy as np
from gym import spaces
import time
import matplotlib.pyplot as plt
OUTPUT_GRAPH = True
N_WORKERS = multiprocessing.cpu_count()
MAX_EP_STEP = 200
MAX_GLOBAL_EP = 2000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.01
LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
U1_block = []
R1_block = []
M2_block = []
D3_block = []

class slicing():
    def __init__(self):
        self.M = 10
        self.lambda_B = 6 * 10 ** (-4)
        self.tau = 10 ** (-3)
        self.sigma_2 = 10 ** (-9)
        self.alpha = 4
        self.B_C = 2000
        self.B_T = 10
        self.B_M_T = 10000000 / 10
        self.P_max = 10
        self.m_C = 3000
        self.L = 1.4
        self.m_1 = 0.001
        self.m_2 = 1
        self.psai = 800
        self.R_1_av = 20000
        self.R_2_av = 20000
        self.N = 3
        self.action_high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     #1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], dtype=np.float64)
        self.action_low =  np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     #0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.float64)
        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float64)
        self.observation_high = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1
             # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
             ], dtype=np.float64)
        self.observation_low = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                         # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         ], dtype=np.float64)
        self.observation_space = spaces.Box(low=self.observation_low, high=self.observation_high, dtype=np.float64)

    def reset(self):
        self.state = np.random.uniform(low=self.observation_low, high=self.observation_high, size=(90,))
        return self.state

    def step(self, state1, ii, Action1, Action2, P_1, P_2, p_3):
        self.K1 = 30
        self.K2 = 30
        self.K3 = 30
        self.K = self.K1 + self.K2 + self.K3
        self.Q3 = np.zeros(self.K3)
        HT1 = state1[:self.K1]
        HT2 = state1[self.K1:self.K1 + self.K2]
        HT3 = state1[-self.K3:]
        Action_1 = np.round(Action1)
        Action_2 = np.round(Action2)

        for k1 in range(self.K1):
            for k2 in range(self.K2):
                for m in range(self.M):
                    if Action_1[k1][m] > 0.0:
                        if Action_2[k2][m] > 0.0:
                            Action_2[k1][m] = 0.0
        agent1_action1 = Action_1[0:int(30/self.N), :]
        agent2_action1 = Action_1[int(30/self.N):2 * int(30/self.N), :]
        agent3_action1 = Action_1[2*int(30/self.N):3*int(30/self.N), :]
        agent1_action2 = Action_2[0:int(30/self.N), :]
        agent2_action2 = Action_2[int(30/self.N):2*int(30/self.N), :]
        agent3_action2 = Action_2[2*int(30/self.N):3*int(30/self.N), :]
        agent1_action = np.append(agent1_action1, agent1_action2, axis=0)
        agent2_action = np.append(agent2_action1, agent2_action2, axis=0)
        agent3_action = np.append(agent3_action1, agent3_action2, axis=0)
        self.K_B=(self.K1+30)/self.N
        for i in range(self.M):
            for x in range(int(self.K_B)):
                for y in range(x + 1, int(self.K_B)):
                    if agent1_action[x][i] == agent1_action[y][i]:
                        if agent1_action[x][i] != 0.0 and agent1_action[y][i] != 0.0:
                            agent1_action[y][i] = 0.0

        for ii in range(self.M):
            for xx in range(int(self.K_B)):
                for yy in range(xx + 1, int(self.K_B)):
                    if agent2_action[yy][ii] == agent2_action[xx][ii]:
                        if agent2_action[yy][ii] != 0.0 and agent2_action[xx][ii] != 0.0:
                            agent2_action[yy][ii] = 0.0
        for iii in range(self.M):
            for xxx in range(int(self.K_B)):
                for yyy in range(xxx + 1, int(self.K_B)):
                    if agent3_action[yyy][iii] == agent3_action[xxx][iii]:
                        if agent3_action[yyy][iii] != 0.0 and agent3_action[xxx][iii] != 0.0:
                            agent3_action[yyy][iii] = 0.0

        agent_action1 = np.concatenate([agent1_action[0:int(self.K1/self.N), :], agent2_action[0:int(self.K1/self.N), :], agent3_action[0:int(self.K1/self.N), :]], axis=0)
        agent_action2 = np.concatenate([agent1_action[int(30/self.N):int(30/self.N)+int(self.K2/self.N), :], agent2_action[int(30/self.N):int(30/self.N)+int(self.K2/self.N), :], agent3_action[int(30/self.N):int(30/self.N+self.K2/self.N), :]], axis=0)
        R1 = np.zeros(self.K1)
        I1 = np.zeros(self.K1)
        for m in range(self.M):
            for k in range(self.K1):
                for kk in range(k + 10, self.K1):
                    if agent_action1[k][m] == agent_action1[kk][m]:
                        if agent_action1[k][m]!= 0.0 and agent_action1[kk][m]!= 0.0:
                            I1[k] = I1[k] + P_1[kk] * HT1[kk]

        for k in range(self.K1):

            R1[k] = sum(agent_action1[k][:]) * self.B_M_T * np.log(1 + P_1[k] * HT1[k]/(I1[k]+self.sigma_2))

        R1_sum=sum(R1)
        count = np.zeros(self.K2)
        for k in range(self.K2):
            con = sum(agent_action2[k,:])
            if con > 0:
                count[k] = 1
        M_C = sum(count)/self.K2
        I3 = np.zeros(self.K3)
        for m in range(self.M):
            for k in range(self.K3):
                for kk in range(k + int(self.K3/self.N), self.K3):
                    if agent_action2[k][m] == agent_action2[kk][m]:
                        if agent_action2[k][m]!= 0.0 and agent_action2[kk][m]!= 0.0:
                            I3[k] =I3[k] + p_3[kk] * HT3[kk]

        R3 = np.zeros(self.K3)
        lambdaS3 = 30
        A3 = np.random.poisson(lambdaS3, size=self.K3)
        Tau = np.zeros(self.K3)

        for L_T in range(self.K3):
            R3[L_T] = sum(agent_action2[L_T][:]) * self.B_M_T * np.log(
                1 + p_3[L_T] * HT3[L_T] / (I3[L_T] + P_2[L_T] * HT2[L_T] + self.sigma_2))
            if R3[L_T] > self.R_1_av:
                Z1 = self.Q3[L_T] - R3[L_T] * self.tau + A3[L_T]
                self.Q3[L_T] = max(Z1, 0)
            else:
                Z2 = self.Q3[L_T] - R3[L_T] * self.tau
                self.Q3[L_T] = max(Z2, 0)
            Tau[L_T] = R3[L_T] / A3[L_T]
        R3_sum=sum(R3)
        P_re=self.L /R3_sum
        delay = sum(Tau)/self.K3
        Dely_sum=(1-P_re) * delay
        m1 = 1
        m2 = 2
        m3 = 1
        U = m1 * R1_sum + m2 * M_C - m3 * Dely_sum
        return U, R1_sum, M_C, Dely_sum


env = slicing()
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]


class ACNet(object):
    def __init__(self, scope, globalAC=None):
        if scope == GLOBAL_NET_SCOPE:  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:  # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-4

                normal_dist = tf.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), A_BOUND[0], A_BOUND[1])
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})


class Worker(object):
    def __init__(self, name, globalAC):
        self.env = slicing()
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP, D3_block,U1_block,R1_block,M2_block
        total_step = 1

        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            ep_R1 = 0
            ep_U1 = 0
            ep_M2 = 0
            ep_D3 = 0
            M1 = 30
            M2 = 30
            M3 = 30
            MM = M1 + M2 + M3
            for ep_t in range(MAX_EP_STEP):
                # if self.name == 'W_0':
                #     self.env.render()

                aaction = self.AC.choose_action(s)
                Action = aaction[0:60 * 12] * 0.5 + 0.5
                action = abs(aaction[60 * 12:60 * 12 + MM])

                Action1 = Action.reshape(60, 12)

                A_1 = Action1[0:30, :]
                A_2 = Action1[30:60, :]

                a_1 = np.transpose(action[0:M1])
                a_2 = np.transpose(action[M1:M1 + M2])
                a_3 = np.transpose(action[M1 + M2:MM])

                U_1, R_1, M_2, Delay3 = self.env.step(s, GLOBAL_EP, A_1, A_2, a_1, a_2, a_3)
                done = True if ep_t == MAX_EP_STEP - 1 else False
                r = U_1
                ep_r += r
                ep_U1 += U_1
                ep_R1 += R_1
                ep_M2 += M_2
                ep_D3 += Delay3
                buffer_s.append(s)
                buffer_a.append(aaction)
                buffer_r.append(r)  # normalize

                s_ = self.env.reset()

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
                        buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_

                total_step += 1

                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)

                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)

                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                    )
                    GLOBAL_EP += 1
                    D3 = ep_D3 / MAX_EP_STEP
                    U1 = ep_U1 / MAX_EP_STEP
                    M2 = ep_M2 / MAX_EP_STEP
                    R1 = ep_R1 / MAX_EP_STEP
                    D3_block.append(D3)
                    U1_block.append(U1)
                    R1_block.append(R1)
                    M2_block.append(M2)
                    break



if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i  # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
    end = time.clock()
    runtime = end - start
    print('runtime', runtime)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('D3')
    plt.show()
