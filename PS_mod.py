
#Improved Profit Sharing

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
        #print("q-values: ",q_values)

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
        #print("q-values: ",q_values)

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
                            f = os.path.join(os.path.dirname(__file__),"./q_table/7x7nv/psm_Boltz_a1.csv")
                        else:
                            f = os.path.join(os.path.dirname(__file__),"./q_table/7x7nv/psm_Boltz_a2.csv")
                
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
                                f = os.path.join(os.path.dirname(__file__),"./q_table/3x3/psm_Boltz_a1.csv")
                            elif map_size == "5x5":
                                f = os.path.join(os.path.dirname(__file__),"./q_table/5x5/psm_Boltz_a1.csv")
                            else:
                                f = os.path.join(os.path.dirname(__file__),"./q_table/7x7/psm_Boltz_a1.csv")
                        else:
                            if map_size == "3x3":
                                f = os.path.join(os.path.dirname(__file__),"./q_table/3x3/psm_Boltz_a2.csv")
                            elif map_size == "5x5":
                                f = os.path.join(os.path.dirname(__file__),"./q_table/5x5/psm_Boltz_a2.csv")
                            else:
                                f = os.path.join(os.path.dirname(__file__),"./q_table/7x7/psm_Boltz_a2.csv")
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

            if self.p == 1:
                #profit sharing mod
                if reward > 0:
                    self.learn(reward)
                elif reward <= -8:
                    self.s_rule = []
                    self.a_rule = []
                """
                #即時の負の報酬
                elif reward < 0 and reward > -10:
                    self.learn(reward)
                """
                
    def learn(self,reward):
        #Profit Sharing
        if reward > 0 or reward <= -10:
            is_ok = 0
            is_del = 0
            
            #スキップ可能かどうか全探索
            while(is_ok == 0):
                #スキップ始めの位置
                start = 0
                for num in range(0,len(self.s_rule)):
                    #スキップ終わりの位置
                    end = start

                    #状態と行動のコピー -> リストをコピーしてpopで引っこ抜く
                    temp_state = self.s_rule[num]
                    temp_action = self.a_rule[num]

                    if is_del == 0:
                        #一回も消去していなければ
                        for itr in range(0,len(self.s_rule)-start):
                            state = self.s_rule[itr + start]
                            action = self.a_rule[itr + start]
                            #同じ状態と行動を探索
                            if (start != end and temp_state == state and temp_action == action):
                                #同じ状態、行動を見つけたらそこまでスキップ
                                del self.s_rule[start:end]
                                del self.a_rule[start:end]

                                #繰り返しから離脱
                                is_del = 1
                                break
                            
                            end += 1
                    
                    if is_del == 1:
                        #一回でも消去したなら、forから抜けてwhileへ
                        break
                    else:
                        #一回も消去しなかったら、whileから抜ける
                        is_ok = 1
                    
                    start += 1

                if start >= len(self.s_rule):
                    break
            
            #逆順
            self.s_rule.reverse()
            self.a_rule.reverse()

            mag = 1
            next_q_values = "None"

            for state,action in zip(self.s_rule,self.a_rule):
                #現在のQ値
                q = self.q_values[state][action]

                #次の期待値を求める
                if next_q_values == "None":
                    expected_reward = reward
                else:
                    q_list = next_q_values
                    fq_list = []
                    sum_q = 0
                    for _idx,obj in enumerate(q_list):
                        fq_list.append(float(obj))
                        sum_q += float(obj)
                    
                    #期待利得
                    expected_reward = 0
                    for idx,obj in enumerate(fq_list):
                        try:
                            #行動選択確率
                            temp = obj/sum_q
                        except:
                            temp = 0
                        
                        expected_reward += fq_list[idx]*temp

                #更新式
                #self.q_values[state][action] = q + (0.2**mag)*(reward)
                #(0.8*reward+0.2*
                self.q_values[state][action] = q + (0.2**mag)*(reward + 0.2*expected_reward)
                next_q_values = self.q_values[state]
                mag += 1
        
            #ルールの初期化
            self.s_rule = []
            self.a_rule = []
            #print("Learn")

        else:
            #衝突時の罰
            mag = len(self.s_rule)
            q = self.q_values[self.previous_state][self.previous_action_id]
            self.q_values[self.previous_state][self.previous_action_id] = q + (0.2**mag)*reward
    
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
                    file = os.path.join(os.path.dirname(__file__),"./q_table/7x7nv/psm_Boltz_a1.csv")
                else:
                    file = os.path.join(os.path.dirname(__file__),"./q_table/7x7nv/psm_Boltz_a2.csv")
        
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
                        file = os.path.join(os.path.dirname(__file__),"./q_table/3x3/psm_Boltz_a1.csv")
                    elif self.map_size == "5x5":
                        file = os.path.join(os.path.dirname(__file__),"./q_table/5x5/psm_Boltz_a1.csv")
                    else:
                        file = os.path.join(os.path.dirname(__file__),"./q_table/7x7/psm_Boltz_a1.csv")
                else:
                    if self.map_size == "3x3":
                        file = os.path.join(os.path.dirname(__file__),"./q_table/3x3/psm_Boltz_a2.csv")
                    elif self.map_size == "5x5":
                        file = os.path.join(os.path.dirname(__file__),"./q_table/5x5/psm_Boltz_a2.csv")
                    else:
                        file = os.path.join(os.path.dirname(__file__),"./q_table/7x7/psm_Boltz_a2.csv")
        
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

            
        