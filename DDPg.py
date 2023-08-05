"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

from gym import spaces
import tensorflow as tf
import numpy as np
import gym
import time
import matplotlib.pyplot as plt
start = time.clock()
np.random.seed(1)
tf.set_random_seed(1)


#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 2000
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            # you can try different target replacement strategies
MEMORY_CAPACITY = 5000
BATCH_SIZE = 32

RENDER = False
OUTPUT_GRAPH = True

###############################  Actor  ####################################
class slicing():
    def __init__(self):
        self.M = 10
        self.lambda_B = 6 * 10 ** (-4)
        self.tau = 10 ** (-3)
        self.sigma_2 = 10 ** (-9)
        self.alpha = 4
        self.B_C = 2000
        self.B_T = 10  # 单位是M
        self.B_M_T = 10000000 / 10
        self.P_max = 10
        self.m_C = 3000
        self.L = 1.4  ## uRLLC服从L的二项分布
        self.m_1 = 0.001
        self.m_2 = 1
        self.psai = 800
        self.R_1_av = 20000
        self.R_2_av = 20000
        self.p_T_1 = 0.4  # 虚拟基站1的激活概率/用户请求切片1的概率
        self.p_T_2 = 0.6  # 虚拟基站2的激活概率/用户请求切片2的概率
        # self.K1 = 30
        self.N = 3
        # self.K_B = 20
        # self.K2 = 30
        # self.K3 = self.K2
        # self.K = self.K1 + self.K2 +self.K3
        # self.action1 = np.zeros((self.M, self.K1))  # 切片1子载波分配
        # self.action2 = np.zeros((self.M, self.K2))  #  NOMA子载波分配
        # self.Action1 = np.zeros((self.M, self.K1))  # 切片1功率分配
        # self.Action2 = np.zeros((self.M, self.K2))  # 切片1功率分配
        # self.Action3 = np.zeros((self.M, self.K3))  # 切片1功率分配
        # self.H1 = np.zeros((self.M, self.K1))
        # self.H2 = np.zeros((self.M, self.K2))
        # self.H3 = np.zeros((self.M, self.K3))
        # self.Q3 = np.zeros(self.K3)
        self.action_high = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
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
             # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
             6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
             6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            dtype=np.float64)
        self.action_low = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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
             # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            dtype=np.float64)
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

        lambdaS3 = 50 * 100 * 1000
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

class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(s, 30, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self, s, noise):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


###############################  Critic  ####################################

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = tf.stop_gradient(a)    # stop critic update flows to actor
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, self.a)[0]   # tensor of gradients of each sample (None, a_dim)

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1


#####################  Memory  ####################

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


env = slicing()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')


sess = tf.Session()

# Create actor and critic.
# They are actually connected to each other, details can be seen in tensorboard or in this picture:
actor = Actor(sess, action_dim, action_bound, LR_A, REPLACEMENT)
critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

sess.run(tf.global_variables_initializer())

M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)


GLOBAL_RUNNING_R= []
GLOBAL_R1 = []
GLOBAL_M2 = []
GLOBAL_D3 = []
for i in range(MAX_EPISODES):

    s = env.reset()
    ep_reward = 0
    ep_R1 = 0
    ep_M2 = 0
    ep_D3 = 0
    M1 = 30
    M2 = 30
    M3 = 30
    MM = M1 + M2 + M3
    for j in range(MAX_EP_STEPS):


        aaction = actor.choose_action(s, 0.3)
        Action = aaction[0:60 * 12] * 0.5 + 0.5
        action = abs(aaction[60 * 12:60 * 12 + MM])

        Action1 = Action.reshape(60, 12)

        A_1 = Action1[0:30, :]
        A_2 = Action1[30:60, :]

        a_1 = np.transpose(action[0:M1])
        a_2 = np.transpose(action[M1:M1 + M2])
        a_3 = np.transpose(action[M1 + M2:MM])

        r, R_1, M_2, Delay3 = env.step(s, i, A_1, A_2, a_1, a_2, a_3)

        s_ = env.reset()
        M.store_transition(s, aaction, r, s_)

        if M.pointer > MEMORY_CAPACITY:
            # decay the action randomness
            b_M = M.sample(BATCH_SIZE)
            b_s = b_M[:, :state_dim]
            b_a = b_M[:, state_dim: state_dim + action_dim]
            b_r = b_M[:, -state_dim - 1: -state_dim]
            b_s_ = b_M[:, -state_dim:]

            critic.learn(b_s, b_a, b_r, b_s_)
            actor.learn(b_s)

        s = s_
        ep_reward += r
        ep_R1 += R_1
        ep_M2 += M_2
        ep_D3 += Delay3
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), )
            if ep_reward > -300:
                RENDER = True
            break
    GLOBAL_RUNNING_R.append(ep_reward/MAX_EP_STEPS)
    GLOBAL_R1.append(ep_R1 / MAX_EP_STEPS)
    GLOBAL_M2.append(ep_M2 / MAX_EP_STEPS)
    GLOBAL_D3.append(ep_D3 / MAX_EP_STEPS)

np.savetxt('GLOBAL_RUNNING_R.csv', GLOBAL_RUNNING_R, fmt='%f')

plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
plt.xlabel('step')
plt.ylabel('GLOBAL_D3')
plt.show()