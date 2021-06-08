
#Profit Sharing and Q-lambda

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
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

class BoltzmannQPolicy():
    def __init__(self,temp=0,training=1):
        self.temp = temp
        self.training = training
        if self.training == 1:
            self.eps = 0.99
        else:
            self.eps = 0.0
    
    def select_action(self,q_values):
        assert q_values.ndim == 1
        _nb_actions = q_values.shape[0]
        max_value = 0.0
        fq_values = []

        np.set_printoptions(suppress=True)
        #print("q_value: ",q_values)

        for index,obj in enumerate(q_values):
            #型変換
            try:
                fq_values.append(float(obj))
            except:
                print("q_values:",q_values)

            #負の値なら０へ
            if fq_values[index] < 0.0:
                fq_values[index] = 0.0

            #合計
            max_value += fq_values[index]

        #わずかな確率でランダム化
        eps = random.uniform(0.0,1.0)

        if max_value == 0.0 or eps <= self.eps:
            #合計が０ならランダム
            action = random.choice([0,1,2,3,4])
        else:
            if self.eps > 0.1:
                self.eps = self.eps*self.eps

            for index,_obj in enumerate(fq_values):
                try:
                    fq_values[index] = fq_values[index] / max_value
                except:
                    fq_values[index] = 0.0

            rand = random.uniform(0.0,1.0)

            if rand <= fq_values[0]:
                action = 0
            elif rand <= fq_values[0]+fq_values[1] and rand > fq_values[0]:
                action = 1
            elif rand <= fq_values[0]+fq_values[1]+fq_values[2] and rand > fq_values[0]+fq_values[1]:
                action = 2
            elif rand <= fq_values[0]+fq_values[1]+fq_values[2]+fq_values[3] and rand > fq_values[0]+fq_values[1]+fq_values[2]:
                action = 3
            else:
                action = 4 
        
        return action

    def select_greedy_action(self,q_values):
        assert q_values.ndim == 1
        _nb_actions = q_values.shape[0]
        max_value = 0.0
        fq_values = []

        for index,obj in enumerate(q_values):
            #型変換
            fq_values.append(float(obj))

            #負の値なら０へ
            if fq_values[index] < 0.0:
                fq_values[index] = 0.0

            #合計
            max_value += fq_values[index]       

        if max_value == 0.0:
            #合計が０ならランダム
            action = random.choice([0,1,2,3,4])
        else:

            for index,_obj in enumerate(fq_values):
                try:
                    fq_values[index] = fq_values[index] / max_value
                except:
                    fq_values[index] = 0.0

            rand = random.uniform(0.0,1.0)

            if rand <= fq_values[0]:
                action = 0
            elif rand <= fq_values[0]+fq_values[1] and rand > fq_values[0]:
                action = 1
            elif rand <= fq_values[0]+fq_values[1]+fq_values[2] and rand > fq_values[0]+fq_values[1]:
                action = 2
            elif rand <= fq_values[0]+fq_values[1]+fq_values[2]+fq_values[3] and rand > fq_values[0]+fq_values[1]+fq_values[2]:
                action = 3
            else:
                action = 4 
        
        return action

class QlearningAgent:

    def __init__(self,aid=0,alpha=.2,policy=None,
                 gamma=.99,actions=None,observation=None,training=True,
                 agent_num=0,map_size="5x5",mode=0,non_vision=0):
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

        self.p = mode  #１：ボルツマン、０：グリーディ

        self.s_rule = []
        self.a_rule = []

        if self.training:
            self.q_values = self._init_q_values()
        else:
            #学習済みのQテーブルを使うなら
            try:
                if self.non_vision == 1:
                    if mode == 0:
                        if agent_num == 0:
                            f = os.path.join(os.path.dirname(__file__),"./q_table/7x7nv/ql_trace_EpsG_a1.csv")
                        else:
                            f = os.path.join(os.path.dirname(__file__),"./q_table/7x7nv/ql_trace_EpsG_a2.csv")
                    else:
                        if agent_num == 0:
                            f = os.path.join(os.path.dirname(__file__),"./q_table/7x7nv/ps_Boltz_a1.csv")
                        else:
                            f = os.path.join(os.path.dirname(__file__),"./q_table/7x7nv/ps_Boltz_a2.csv")
                
                else:
                    if mode == 0:
                        if agent_num == 0:
                            if map_size == "3x3":
                                f = os.path.join(os.path.dirname(__file__),"./q_table/3x3/ql_trace_EpsG_a1.csv")
                            elif map_size == "5x5":
                                f = os.path.join(os.path.dirname(__file__),"./q_table/5x5/ql_trace_EpsG_a1.csv")
                            else:
                                f = os.path.join(os.path.dirname(__file__),"./q_table/7x7/ql_trace_EpsG_a1.csv")
                        else:
                            if map_size == "3x3":
                                f = os.path.join(os.path.dirname(__file__),"./q_table/3x3/ql_trace_EpsG_a2.csv")
                            elif map_size == "5x5":
                                f = os.path.join(os.path.dirname(__file__),"./q_table/5x5/ql_trace_EpsG_a2.csv")
                            else:
                                f = os.path.join(os.path.dirname(__file__),"./q_table/7x7/ql_trace_EpsG_a2.csv")
                        
                    else:
                        if agent_num == 0:
                            if map_size == "3x3":
                                f = os.path.join(os.path.dirname(__file__),"./q_table/3x3/ps_Boltz_a1.csv")
                            elif map_size == "5x5":
                                f = os.path.join(os.path.dirname(__file__),"./q_table/5x5/ps_Boltz_a1.csv")
                            else:
                                f = os.path.join(os.path.dirname(__file__),"./q_table/7x7/ps_Boltz_a1.csv")
                        else:
                            if map_size == "3x3":
                                f = os.path.join(os.path.dirname(__file__),"./q_table/3x3/ps_Boltz_a2.csv")
                            elif map_size == "5x5":
                                f = os.path.join(os.path.dirname(__file__),"./q_table/5x5/ps_Boltz_a2.csv")
                            else:
                                f = os.path.join(os.path.dirname(__file__),"./q_table/7x7/ps_Boltz_a2.csv")
            except:
                print("no file")

            self.q_values = self.read_q(f)
    
    def _init_q_values(self):
        #q-tableの更新
        q_values = {self.state: np.repeat(0.0, len(self.actions))}
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

        #始めて訪れる状態であれば
        if next_state not in self.q_values:
            self.q_values[next_state] = np.repeat(0.0, len(self.actions))
        
        self.previous_state = copy.deepcopy(self.state)
        self.state = next_state

        if self.training and reward is not None:
            #ルールに格納
            self.s_rule.append(self.previous_state)
            self.a_rule.append(self.previous_action_id)
            self.reward_history.append(reward)

            if self.p == 0:
                #q-lambda
                self.learn(reward)
            else:
                #profit sharing
                if reward > 0:
                    self.learn2(reward)
                elif reward <= -8:
                    self.s_rule = []
                    self.a_rule = []
    
    def learn(self,reward):
        #q-lambda
        if reward > 0:
            #正の報酬を得られた時、一括強化
            mag = len(self.s_rule)
            for state,action in zip(self.s_rule,self.a_rule):
                q = self.q_values[state][action]
                max_q = max(self.q_values[state])
                self.q_values[state][action] = q + (self.alpha * ((0.9**mag)*reward + (self.gamma * max_q) - q))
                mag -= 1
            #ルールの初期化
            self.s_rule = []
            self.a_rule = []
            #print("q_table:",self.q_values)

        else:          
            #Q-valueの更新
            mag = len(self.s_rule)
            q = self.q_values[self.previous_state][self.previous_action_id]
            max_q = max(self.q_values[self.state])
            #Q(s,a) = Q(s,a)+alpha*(r+gamma*maxQ(s')-Q(s,a))
            self.q_values[self.previous_state][self.previous_action_id] = q +(self.alpha * ((0.9**mag)*reward + (self.gamma * max_q) - q))
    
    def learn2(self,reward):
        #Profit Sharing
        #ボルツマン用
        #正の報酬を得られた時、一括強化       
        #Q値ではなく報酬で
        #(0.5**mag),mag = len(self.s_rule)+1
        if reward > 0:
            mag = len(self.s_rule)
            for state,action in zip(self.s_rule,self.a_rule):
                q = self.q_values[state][action]
                self.q_values[state][action] = q + (0.2**mag)*reward
                mag -= 1
        
            #ルールの初期化
            self.s_rule = []
            self.a_rule = []
            #print("Learn")

        else:
            #衝突時の罰
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
            if self.p == 0:
                if agent_num == 0:
                    file = os.path.join(os.path.dirname(__file__),"./q_table/7x7nv/ql_trace_EpsG_a1.csv")
                else:
                    file = os.path.join(os.path.dirname(__file__),"./q_table/7x7nv/ql_trace_EpsG_a2.csv")
            else:
                if agent_num == 0:
                    file = os.path.join(os.path.dirname(__file__),"./q_table/7x7nv/ps_Boltz_a1.csv")
                else:
                    file = os.path.join(os.path.dirname(__file__),"./q_table/7x7nv/ps_Boltz_a2.csv")
        
        else:
            if self.p == 0:
                if agent_num == 0:
                    if self.map_size == "3x3":
                        file = os.path.join(os.path.dirname(__file__),"./q_table/3x3/ql_trace_EpsG_a1.csv")
                    elif self.map_size == "5x5":
                        file = os.path.join(os.path.dirname(__file__),"./q_table/5x5/ql_trace_EpsG_a1.csv")
                    else:
                        file = os.path.join(os.path.dirname(__file__),"./q_table/7x7/ql_trace_EpsG_a1.csv")
                else:
                    if self.map_size == "3x3":
                        file = os.path.join(os.path.dirname(__file__),"./q_table/3x3/ql_trace_EpsG_a2.csv")
                    elif self.map_size == "5x5":
                        file = os.path.join(os.path.dirname(__file__),"./q_table/5x5/ql_trace_EpsG_a2.csv")
                    else:
                        file = os.path.join(os.path.dirname(__file__),"./q_table/7x7/ql_trace_EpsG_a2.csv")
                
            else:
                if agent_num == 0:
                    if self.map_size == "3x3":
                        file = os.path.join(os.path.dirname(__file__),"./q_table/3x3/ps_Boltz_a1.csv")
                    elif self.map_size == "5x5":
                        file = os.path.join(os.path.dirname(__file__),"./q_table/5x5/ps_Boltz_a1.csv")
                    else:
                        file = os.path.join(os.path.dirname(__file__),"./q_table/7x7/ps_Boltz_a1.csv")
                else:
                    if self.map_size == "3x3":
                        file = os.path.join(os.path.dirname(__file__),"./q_table/3x3/ps_Boltz_a2.csv")
                    elif self.map_size == "5x5":
                        file = os.path.join(os.path.dirname(__file__),"./q_table/5x5/ps_Boltz_a2.csv")
                    else:
                        file = os.path.join(os.path.dirname(__file__),"./q_table/7x7/ps_Boltz_a2.csv")
        
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

            
        