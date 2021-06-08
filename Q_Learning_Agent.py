
#Q-Learning

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import csv
import os

class EpsGreedyQPolicy:
    #epsilon-greedy

    def __init__(self, epsilon=.05, decay_rate=1):
        self.epsilon = epsilon
        self.decay_rate = decay_rate
    
    def select_action(self, q_values):
        #行動選択
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        #ε-greedy行動
        if np.random.uniform() < self.epsilon:
            action = np.random.random_integers(0, nb_actions - 1)
        else:
            action = np.argmax(q_values)
        
        self.decay_eps_rate()
        return action
    
    def decay_eps_rate(self):
        self.epsilon = self.epsilon*self.decay_rate
        if self.epsilon < .01:
            self.epsilon = .01
    
    def select_greedy_action(self, q_values):
        assert q_values.ndim == 1
        action = np.argmax(q_values)
        #判定する

        return action

class QlearningAgent:

    def __init__(self,aid=0,alpha=.2,policy=None,
                 gamma=.99,actions=None,observation=None,training=True,
                 agent_num=0,map_size="5x5",non_vision=0):
        self.aid = aid
        self.alpha = alpha
        self.gamma = gamma
        self.policy = policy
        self.reward_history = []
        self.actions = actions
        self.state = str(observation)
        self.ini_state = str(observation)
        self.previous_state = None
        self.previous_action_id = None
        self.training = training            #trainingかどうか
        self.qtemp = self._init_q_values()  #Q_valueを一時的に格納する変数
        self.map_size = map_size

        self.previous_action = 0    #ダミー変数
        self.non_vision = non_vision

        if self.training:
            self.q_values = self._init_q_values()
        else:
            #学習済みのQテーブルを使うなら
            try:
                if non_vision == 1:
                    if agent_num == 0:
                        f = os.path.join(os.path.dirname(__file__),"./q_table/7x7nv/ql_EpsG_a1.csv")
                    else:
                        f = os.path.join(os.path.dirname(__file__),"./q_table/7x7nv/ql_EpsG_a2.csv")

                else:
                    if agent_num == 0:
                        if map_size == "3x3":
                            f = os.path.join(os.path.dirname(__file__),"./q_table/3x3/ql_EpsG_a1.csv")
                        elif map_size == "5x5":
                            f = os.path.join(os.path.dirname(__file__),"./q_table/5x5/ql_EpsG_a1.csv")
                        else:
                            f = os.path.join(os.path.dirname(__file__),"./q_table/7x7/ql_EpsG_a1.csv")
                    else:
                        if map_size == "3x3":
                            f = os.path.join(os.path.dirname(__file__),"./q_table/3x3/ql_EpsG_a2.csv")
                        elif map_size == "5x5":
                            f = os.path.join(os.path.dirname(__file__),"./q_table/5x5/ql_EpsG_a2.csv")
                        else:
                            f = os.path.join(os.path.dirname(__file__),"./q_table/7x7/ql_EpsG_a2.csv")
            except:
                print("no file")

            self.q_values = self.read_q(f)
    
    def _init_q_values(self):
        #q-tableの更新
        q_values = {self.state: np.repeat(0., len(self.actions))}
        return q_values
    
    def init_state(self):
        #状態の初期化
        self.previous_state = copy.deepcopy(self.ini_state)
        self.state = copy.deepcopy(self.ini_state)
        return self.state
    
    def act(self,q_values=None):
        self.qtemp = self.q_values[self.state]

        if self.training:
            action_id = self.policy.select_action(self.qtemp)
        else:
            #最善手を選ぶため行動が変更されない→無限ループ
            action_id = self.policy.select_greedy_action(self.qtemp)
        
        self.previous_action_id = action_id
        action = self.actions[action_id]
        return action
    
    def react(self,q_values=None):
        max_id = np.argmax(self.qtemp)
        self.qtemp[max_id] = -100

        action_id = self.policy.select_greedy_action(self.qtemp)
        
        self.previous_action_id = action_id
        action = self.actions[action_id]
        return action
    
    def observe(self, next_state, reward=None,reward_o=None,opponent_action=0,is_learn=True):
        #次の状態と報酬の観測
        next_state = str(next_state)
        #print(next_state)
        #始めて訪れる状態であれば
        if next_state not in self.q_values:
            self.q_values[next_state] = np.repeat(0.0, len(self.actions))
        
        self.previous_state = copy.deepcopy(self.state)
        self.state = next_state

        if self.training and reward is not None:
            self.reward_history.append(reward)
            self.learn(reward)
    
    def learn(self,reward):
        #Q-valueの更新
        q = self.q_values[self.previous_state][self.previous_action_id]
        max_q = max(self.q_values[self.state])
        #Q(s,a) = Q(s,a)+alpha*(r+gamma*maxQ(s')-Q(s,a))
        self.q_values[self.previous_state][self.previous_action_id] = q + (self.alpha * (reward + (self.gamma * max_q) - q))
        
    def save_q(self,agent_num):
        #Q-tableの保存
        q_dict = self.q_values   #辞書型
        q_row = {}
        #print(q_dict)

        if self.non_vision == 1:
            if agent_num == 0:
                file = os.path.join(os.path.dirname(__file__),"./q_table/7x7nv/rand_ql_EpsG_a1.csv")
            else:
                file = os.path.join(os.path.dirname(__file__),"./q_table/7x7nv/ql_EpsG_a2.csv")
        else:        
            if agent_num == 0:
                if self.map_size == "3x3":
                    file = os.path.join(os.path.dirname(__file__),"./q_table/3x3/ql_EpsG_a1.csv")
                elif self.map_size == "5x5":
                    file = os.path.join(os.path.dirname(__file__),"./q_table/5x5/ql_EpsG_a1.csv")
                else:
                    file = os.path.join(os.path.dirname(__file__),"./q_table/7x7/ql_EpsG_a1.csv")
            else:
                if self.map_size == "3x3":
                    file = os.path.join(os.path.dirname(__file__),"./q_table/3x3/ql_EpsG_a2.csv")
                elif self.map_size == "5x5":
                    file = os.path.join(os.path.dirname(__file__),"./q_table/5x5/ql_EpsG_a2.csv")
                else:
                    file = os.path.join(os.path.dirname(__file__),"./q_table/7x7/ql_EpsG_a2.csv")
        
        with open(file,'w') as f:
            writer = csv.DictWriter(f,fieldnames=q_dict.keys(),delimiter=",",quotechar='"')
            writer.writeheader()

            k1 = list(q_dict.keys())[0]
            length = len(q_dict[k1])

            for i in range(length):
                for k, vs in q_dict.items():
                    q_row[k] = vs[i]
                
                writer.writerow(q_row)

    def read_q(self,file):
        #Q-tableの読み込み
        with open(file,newline = "") as f:
            read_dict = csv.DictReader(f,delimiter=",", quotechar='"')
            ks = read_dict.fieldnames
            return_dict = {k: [] for k in ks}

            for row in read_dict:
                for k, v in row.items():
                    return_dict[k].append(float(v))
            
            for v in return_dict:
                return_dict[v] = np.array(return_dict[v])
            
        #print(return_dict)
        return return_dict
    
    def print_q(self):
        for s,q in self.q_values.items():
            print("state: "+str(s)+"  q_values: "+str(q))