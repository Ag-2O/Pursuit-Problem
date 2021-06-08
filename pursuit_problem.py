
#追跡問題

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import Q_Learning_Agent as qla
import itertools
import random
import sys
import time

FILED_TYPE = {
    "N" : 0,
    "W" : 1
}

ACTIONS = {
    "UP" : 0,
    "DOWN" : 1,
    "LEFT" : 2,
    "RIGHT" : 3,
    "STAY" : 4
}

class Prey:
    def __init__(self,pid):
        self.pid = pid
        self.is_captured = False
    
    def act(self,preys_pos,agents_pos):
        #Preyの行動
        choice_flag = 0             #探索フラグ
        px,py = preys_pos[0]        #preyの位置
        rng = [-1,0,1]              #探索範囲
        self.actions = [0,1,2,3]    #行動範囲
        for x,y in itertools.product(rng,rng):
            to_py = py + y
            to_px = px + x
            to_pr = (to_px,to_py)
            #その位置にagentsがいたら
            if (to_pr == agents_pos[0] or to_pr == agents_pos[1]):
                #いる方向への行動を削除
                if y>0 and 1 in self.actions:
                    self.actions.remove(1)
                if y<0 and 0 in self.actions:
                    self.actions.remove(0)
                if x>0 and 3 in self.actions:
                    self.actions.remove(3)
                if x<0 and 2 in self.actions:
                    self.actions.remove(2)
            
                choice_flag = 1

        if choice_flag == 1:
            if self.actions == []:
                self.actions.append(4)
            #print(self.actions)
            action = random.choice(self.actions)
            #print("Prey choice :",action)
            return action
        else:
            #探索してない時はランダム
            rand = np.random.randint(5)
            #print("Prey random_choice :",rand)
            return rand
    
    def react(self):
        #行動失敗時
        if self.actions == []:
            rand = np.random.randint(5)
            #print("Prey random_choice :",rand)
            return rand
        else:
            self.actions.append(4)
            action = random.choice(self.actions)
            #print("Prey choice :",action)
            return action
    
    def random_act(self):
        return np.random.randint(5)
    
    def stay_act(self):
        return 4

    def reset(self):
        self.is_captured = False

class PredatorsPursuitGame:
    def __init__(self,agents,is_evaluate=0,map_size="5x5",prey_level="Random"):

        if map_size == "5x5":
            self.map = [[0,0,0,0,0],
                        [0,0,0,0,0],
                        [0,0,0,0,0],
                        [0,0,0,0,0],
                        [0,0,0,0,0]]

        elif map_size == "3x3":
            self.map = [[0,0,0],
                        [0,0,0],
                        [0,0,0]]
        
        elif map_size == "7x7":
            self.map = [[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0]]
        
        self.prey_level = prey_level
        self.map_size = map_size
        self.agents = agents
        self.agents_pos = {}        #辞書型

        #通常
        for agent in self.agents:
            pos = self._init_pos(self.agents_pos.values())
            self.agents_pos[agent.aid] = pos

        self.prey = Prey(0)
        self.preys = [self.prey]    #辞書型

        pos = self._init_pos(self.agents_pos.values())
        self.preys_pos = {0:pos}

    def step(self,actions,nb_step):
        #全エージェントの行動の実行
        #状態、報酬、ゴールしたかどうかをreturn
        is_collision = [0,0]

        #ハンターの移動 aid = agent id
        for aid, action in enumerate(actions):
            x, y = copy.deepcopy(self.agents_pos[aid])
            #deepcopy : 配列の中身(内部配列)を異なるIDにするため
            to_x, to_y, is_collision[aid] = self.move_agent(x,y,action)
            self.agents_pos[aid] = (to_x,to_y)
        
        #preyの行動 pid = prey id
        for _pid, prey in enumerate(self.preys):
            to_x, to_y = self.move_prey(prey)
            self.preys_pos[prey.pid] = (to_x,to_y)
        
        #終了判定(すべてのハンターがpreyに隣接しているか)
        is_capture = self._is_capture()
        is_terminal = self._is_terminal(nb_step)
        r1 = self._compute_reward(is_capture,is_collision[0],actions[0],nb_step)
        r2 = self._compute_reward(is_capture,is_collision[1],actions[1],nb_step)

        obss = {}
        for agent in self.agents:
            obs = self.create_observation(agent)
            obss[agent.aid] = obs
        
        return obss, r1, r2, is_terminal
    
    def _init_pos(self,poss=[]):
        #被らないposデータの生成
        x = np.random.randint(0,len(self.map[0]))
        y = np.random.randint(0,len(self.map))

        while(x,y) in poss:
            x = np.random.randint(0,len(self.map[0]))
            y = np.random.randint(0,len(self.map))
        
        return x,y
    
    def move(self,x,y,action):
        to_x = copy.deepcopy(x)
        to_y = copy.deepcopy(y)

        if action == ACTIONS["UP"]:
            to_y += -1
        elif action == ACTIONS["DOWN"]:
            to_y += 1
        elif action == ACTIONS["LEFT"]:
            to_x += -1
        elif action == ACTIONS["RIGHT"]:
            to_x += 1
        
        return to_x, to_y
    
    def move_agent(self,x,y,action):
        is_collision = False
        to_x,to_y = copy.deepcopy(x), copy.deepcopy(y)
        if self._is_possible_action(x,y,action):
            to_x,to_y = self.move(x,y,action)
        else:
            is_collision = True
        #衝突だったらそのままの位置
        return to_x, to_y, is_collision
    
    def move_prey(self,prey):
        x,y = copy.deepcopy(self.preys_pos[prey.pid])
        if self.prey_level == "Random":
            action = prey.random_act()
        elif self.prey_level == "1-Square-Search":
            action = prey.act(self.preys_pos,self.agents_pos)
        elif self.prey_level == "No-Move":
            action = prey.stay_act()

        while self._is_possible_action(x,y,action) is False:
            if self.prey_level == "Random":
                action = prey.random_act()
            elif self.prey_level == "1-Square-Search":
                action = prey.react()
            elif self.prey_level == "No-Move":
                action = prey.stay_act()
        
        to_x, to_y = self.move(x, y, action)

        return to_x, to_y
    
    def in_map(self,agent_id,action):
        #実行可能な行動かどうかの判定
        x,y = self.agents_pos[agent_id] #位置情報
        to_x = copy.deepcopy(x)
        to_y = copy.deepcopy(y)

        if action == ACTIONS["STAY"]:
            return True
        else:
            to_x, to_y = self.move(to_x, to_y, action)
            
            if len(self.map) <= to_y or 0 > to_y:
                return False
            elif len(self.map[0]) <= to_x or 0 > to_x:
                return False
                
        return True
    
    def create_observation(self,agent):
        #観測情報の生成
        obs = [self.agents_pos[agent.aid]]
        for agent2 in self.agents:
            if agent2 is not agent:
                obs.append(self.agents_pos[agent2.aid])
        
        for prey in self.preys:
            obs.append(self.preys_pos[prey.pid])
        
        return obs
    
    def _is_capture(self):
        #preyを捕まえたかどうかの判定

        is_capture = False

        for prey in self.preys:
            if self._check_adjacent(prey):
                is_capture = True
        
        return is_capture
    
    def _check_adjacent(self,prey):
        #preyの隣接しているところにエージェントはいるかどうかの判定
        nb_adjacent_agents = 0
        #print("prey_pos",self.preys_pos[prey.pid])
        for action in np.arange(0,4):
            to_x, to_y = self.preys_pos[prey.pid]
            to_x, to_y = self.move(to_x,to_y, action)

            if (to_x, to_y) in self.agents_pos.values():
                nb_adjacent_agents += 1
                #print("nb_adjacent_agents",nb_adjacent_agents)
            
        if nb_adjacent_agents >= 2:
            prey.is_captured = True
            return True
        else:
            return False
    
    def _is_terminal(self,nb_step):
        #すべてのpreyを捕まえたかどうかの確認
        is_terminal = True

        #各preyの隣接
        for prey in self.preys:
            if prey.is_captured is False:
                is_terminal = False
        
        if nb_step == 400:
            is_terminal = True
        
        #print("captured: ",is_terminal)
        
        return is_terminal
    
    def _is_wall(self,x,y):
        #x,yが壁化そうかの確認
        if self.map[x][y] == FILED_TYPE["W"]:
            return True
        else:
            return False
    
    def _is_possible_action(self,x,y,action):
        #実行可能な行動かどうかの判定
        if action == ACTIONS["STAY"]:
            return True
        else:
            to_x, to_y = self.move(x,y,action)

            if len(self.map) <= to_y or 0 > to_y:
                return False
            elif len(self.map[0]) <= to_x or 0 > to_x:
                return False
            elif self._is_wall(to_x,to_y) or self._is_agent_or_prey(to_x,to_y):
                return False
        
        return True
    
    def _is_agent_or_prey(self,x,y):
        #選択した状態にエージェントもしくはpreyがいるかの確認

        if (x,y) in self.agents_pos.values():
            return True
        elif (x,y) in self.preys_pos.values():
            return True
        
        return False
    
    def _compute_reward(self,is_capture,is_obstacle,action,nb_step):
        #ターン数毎に-1の報酬
        #R = -1
        R = -1

        #捕まえたら
        if is_capture and nb_step < 400:
            return 10
        elif nb_step == 400:
            return -10
        
        #衝突したら
        if is_obstacle:
            R += -2
        
        return R
    
    def reset(self):
        self.agents_pos = {}
        for agent in self.agents:
            pos = self._init_pos(self.agents_pos.values())
            self.agents_pos[agent.aid] = pos
        pos = self._init_pos(self.agents_pos.values())
        self.preys_pos = {0:pos}
        self.prey.reset()

    def print_current_map(self):
        #表示
        current_map = copy.deepcopy(self.map)
        for x, y in self.agents_pos.values():
            current_map[y][x] = "A"

        for x, y in self.preys_pos.values():
            current_map[y][x] = "P"
        
        current_map = np.array(current_map) 

        return current_map
    
    def get_agents_pos(self):
        #エージェントの位置
        #実際の実行はゲーム開始時のみ
        #print("agents_pos: (x,y)")
        #print(self.agents_pos)
        return self.agents_pos
    
    def position_difference(self):
        #agent同士の位置の差を返す
        dict_agents_pos = self.agents_pos
        pos = []

        #for i, _obj in enumerate(dict_agents_pos):
        for i in range(2):
            pos.append(dict_agents_pos[i])
            if i >= 1:
                x_pos_dif = pos[i-1][0] - pos[i][0]
                y_pos_dif = pos[i-1][1] - pos[i][1]

                if x_pos_dif <= 0:
                    x_pos_dif *= -1
                if y_pos_dif <= 0:
                    y_pos_dif *= -1
                
        return abs(x_pos_dif), abs(y_pos_dif)
    
    def search_1_1(self):
        #座標差が(1,1)の時実行
        #agentsの周囲を探索
        #協調状態かどうかを返す

        rng = [-1,0,1]

        #協調状態かどうか
        is_cooperate_state = 0

        for index,_obj in enumerate(self.agents):
            #味方の方向を探索
            direction_agent = [0,0]

            #agentsが入れ替わっても対応できるように
            if index == 0:
                target = 1
            else:
                target = 0

            for x,y in itertools.product(rng,rng):
                to_y = self.agents_pos[index][1] + y
                to_x = self.agents_pos[index][0] + x
                to_r = (to_x,to_y)
                
                #味方の探索
                if (to_r == self.agents_pos[target]):
                    #味方がいる方向を記録
                    if y>0:    #うえ
                        direction_agent[1] = 1
                    if y<0:    #した
                        direction_agent[1] = -1
                    if x>0:    #みぎ
                        direction_agent[0] = 1
                    if x<0:    #ひだり
                        direction_agent[0] = -1
            
            #その方向へ獲物がいるかどうかを探索
            #print("direction_agent",direction_agent)
            if direction_agent != [0,0]:
                to_pos = list(self.agents_pos[index])
                # x軸方向
                while(True):
                    #脱出条件
                    if to_pos[0] > 7 or to_pos[0] < 0:
                        break
                    
                    #敵がいるかどうか
                    if list(self.preys_pos[0]) == to_pos:
                        is_cooperate_state = 1
                        return is_cooperate_state
                    
                    to_pos[0] += direction_agent[0]
                
                to_pos = list(self.agents_pos[index])
                # y軸方向
                while(True):
                    if to_pos[1] > 7 or to_pos[1] < 0:
                        break

                    #敵がいるかどうか
                    if list(self.preys_pos[0]) == to_pos:
                        is_cooperate_state = 1
                        return is_cooperate_state
                    
                    to_pos[1] += direction_agent[1]

        return is_cooperate_state
                    

    def search_2_0(self):
        #座標差が(2,0)(0,2)の時
        px,py = self.preys_pos[0]
        rng = [-1,0,1]

        #協調状態かどうか
        is_cooperate_state = 0

        #agentのフラグ
        is_agents = 0

        for x,y in itertools.product(rng,rng):
            to_py = py + y
            to_px = px + x
            to_pr = (to_px, to_py)

            if (to_pr == self.agents_pos[0] or to_pr == self.agents_pos[1]):
                #agentsがいたら
                if is_agents == 1:
                    #2体いたら
                    is_cooperate_state = 1
                    return is_cooperate_state
                
                is_agents = 1

        return is_cooperate_state
    
    def search_1_0(self):
        #座標差が(1,0)(0,1)の時
        #十字探索
        cross_rng = [(0,1),(0,-1),(1,0),(-1,0)]

        #強調状態かどうか
        is_cooperate_state = 0

        #味方の探索
        for index, _obj in enumerate(self.agents):

            for x,y in cross_rng:
                to_y = self.agents_pos[index][1] + y
                to_x = self.agents_pos[index][0] + x
                to_r = (to_x,to_y)

                if index == 0:
                    target = 1
                else:
                    target = 0
                
                if (to_r == self.agents_pos[target]):
                    #味方のいる方向
                    if y==1 or y==-1:   #たて
                        prey_rng = [(1,0),(-1,0)]
                        for x,y in prey_rng:
                            to_y = self.agents_pos[index][1]
                            to_x = self.agents_pos[index][0] + x
                            to_r = (to_x,to_y)

                            if (to_r == self.preys_pos[0]):
                                is_cooperate_state = 1
                                return is_cooperate_state

                    elif x==1 or x==-1: #よこ
                        prey_rng = [(0,1),(0,-1)]
                        for x,y in prey_rng:
                            to_y = self.agents_pos[index][1] + y
                            to_x = self.agents_pos[index][0]
                            to_r = (to_x,to_y)

                            if (to_r == self.preys_pos[0]):
                                is_cooperate_state = 1
                                return is_cooperate_state
                    else:
                        print("2_2 Search Error")
        return is_cooperate_state

    def search_2_1(self):
        #座標差が(2,1)or(1,2)の時、(2,2)の時
        is_cooperate_state = 0
        is_x = False
        is_y = False
        for index,_obj in enumerate(self.agents):
            #x座標の比較
            if self.preys_pos[0][0] == self.agents_pos[index][0]:
                is_x = True
            #y座標の比較
            if self.preys_pos[0][1] == self.agents_pos[index][1]:
                is_y = True
        
        if is_x == True and is_y == True:
            is_cooperate_state = 1
        
        return is_cooperate_state
