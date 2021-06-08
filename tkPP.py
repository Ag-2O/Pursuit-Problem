
#評価システムのGUI

import sys
import time
import argparse
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
import threading
import subprocess
from subprocess import PIPE
import importlib
from importlib import machinery
from importlib import import_module
import matplotlib.pyplot as plt
import pandas as pd
import copy
import csv
import itertools

import pursuit_problem as pp
import Profit_Sharing as ps
import Q_Learning_Agent as qla
import PS_mod as psm
import Human_Agent as ha

class PursuitGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pusuit_Problem")
        self.geometry("200x120")
        label = tk.Label(self,text="What do you do?")
        label.pack()

        #playmode
        plBtn = tk.Button(self,text = "Playing",command = self.make_plmode)
        plBtn.pack()

        #RLmode
        rlBtn = tk.Button(self,text = "Learning",command = self.make_rlmode)
        rlBtn.pack()

        #Evmode
        EvBtn = tk.Button(self,text = "Evaluation",command = self.make_evmode)
        EvBtn.pack()
    
    #-------------------------------------------------------------------
    #プレイモード

    def make_plmode(self):
        #プレイ用UI
        self.pl = tk.Toplevel()
        self.pl.title("Play")
        self.pl.geometry("{}x{}".format(350,500))
        self.cv = tk.Canvas(self.pl,bg="white",width=350,height=350)
        self.cv.place(x=0,y=0)
        self.tag2pos = {}
        self.z2tag = {}

        self.agents = []
        self.actions = np.arange(5)

        #視野制限
        self.non_vision = 0

        #操作が人間かどうか
        self.HAF = [0,0]

        #スレッド処理の停止
        self.thread_stop = 0

        #Button設定
        self.StartBtn = tk.Button(self.pl,text="Start",command = lambda: self.thread(non_vision=self.non_vision), width=8)
        self.StartBtn.place(x=282,y=410)

        self.ResetBtn = tk.Button(self.pl,text="Reset",command = lambda: self.reset(mode=1), width=8)
        self.ResetBtn.place(x=282,y=440)

        self.configBtn = tk.Button(self.pl,text="config",command=self.make_config, width=8)
        self.configBtn.place(x=282,y=470)

        #移動ログ表示
        mlbl = tk.Label(self.pl,text="action")
        mlbl.place(x=12,y=405)
        self.movelog = tk.Text(self.pl,width=20,height=5)
        self.movelog.insert(1.0,"---log---")
        self.movelog.place(x=10,y=425)

        #textboxを入れるためのFrame
        self.textFrame = tk.Frame(self.pl,width=100,height=100)
        self.textFrame.configure()
        self.textFrame.place(x=155,y=425)

        #スクロールバー
        self.scrollbar = tk.Scrollbar(self.textFrame,orient=tk.VERTICAL,command=self.movelog.yview)
        self.scrollbar.pack(side=tk.RIGHT,fill="y")
        self.scrollbar.config(command=self.movelog.yview)

        #agents
        self.agent1mode = tk.StringVar()
        self.agent2mode = tk.StringVar()
        self.agent1mode.set("Agent1 : ")
        self.agent2mode.set("Agent2 : ")

        self.agent1_label = tk.Label(self.pl,textvariable = self.agent1mode)
        self.agent1_label.place(x=10,y=360)
        self.agent2_label = tk.Label(self.pl,textvariable = self.agent2mode)
        self.agent2_label.place(x=10,y=380)

        #prey
        self.prey_mode = tk.StringVar()
        self.prey_mode.set("Prey : ")

        self.prey_label = tk.Label(self.pl,textvariable = self.prey_mode)
        self.prey_label.place(x=200,y=360)

        #矢印ボタン
        self.var = tk.IntVar()
        self.var.set(5)
        self.UpBtn = tk.Button(self.pl,text="↑",state=tk.DISABLED,command = lambda: self.var.set(0), width=2)
        self.UpBtn.place(x=212,y=410)

        self.DownBtn = tk.Button(self.pl,text="↓",state=tk.DISABLED,command = lambda: self.var.set(1), width=2)
        self.DownBtn.place(x=212,y=470)

        self.LeftBtn = tk.Button(self.pl,text="←",state=tk.DISABLED,command = lambda: self.var.set(2), width=2)
        self.LeftBtn.place(x=182,y=440)

        self.RightBtn = tk.Button(self.pl,text="→",state=tk.DISABLED,command = lambda: self.var.set(3), width=2)
        self.RightBtn.place(x=242,y=440)

        self.WaitBtn = tk.Button(self.pl,text="〇",state=tk.DISABLED,command = lambda: self.var.set(4), width=2)
        self.WaitBtn.place(x=212,y=440)

    def action_button_able(self):
        self.UpBtn['state'] = tk.NORMAL
        self.DownBtn['state'] = tk.NORMAL
        self.LeftBtn['state'] = tk.NORMAL
        self.RightBtn['state'] = tk.NORMAL
        self.WaitBtn['state'] = tk.NORMAL
    
    def action_button_disable(self):
        self.UpBtn['state'] = tk.DISABLED
        self.DownBtn['state'] = tk.DISABLED
        self.LeftBtn['state'] = tk.DISABLED
        self.RightBtn['state'] = tk.DISABLED
        self.WaitBtn['state'] = tk.DISABLED
    
    def make_config(self,is_evaluate=0):
        #設定用UI
        cfg = tk.Toplevel()
        cfg.title("config")
        cfg.geometry("300x165")

        #agent_labelの設定
        label1 = tk.Label(cfg,text="agent1")
        label1.place(x=10,y=9)
        combo1 = ttk.Combobox(cfg,state="readonly")
        combo1["values"] = ("Q-Learning","Q-Lambda","Profit Sharing","Improved Profit Sharing","NashQ-Learning","Human")
        combo1.current(0)
        combo1.place(x=60,y=10)

        label2 = tk.Label(cfg,text="agent2")
        label2.place(x=10,y=34)
        combo2 = ttk.Combobox(cfg,state="readonly")
        combo2["values"] = ("Q-Learning","Q-Lambda","Profit Sharing","Improved Profit Sharing","NashQ-Learning","Human")
        combo2.current(0)
        combo2.place(x=60,y=35)

        MAP_label = tk.Label(cfg,text="Map")
        MAP_label.place(x=10,y=59)
        MAP_combo= ttk.Combobox(cfg,state="readonly",width=10,height=1)
        MAP_combo["value"] = ("3x3","5x5","7x7")
        MAP_combo.current(0)
        MAP_combo.place(x=60,y=60)

        prey_level_label = tk.Label(cfg,text="Prey")
        prey_level_label.place(x=10,y=84)
        prey_combo = ttk.Combobox(cfg,state="readonly",height=1)
        prey_combo["value"] = ("Random","1-Square-Search","No-Move")
        prey_combo.current(0)
        prey_combo.place(x=60,y=85)

        #視野制限モード
        non_vision_label = tk.Label(cfg,text="Visibility")
        non_vision_label.place(x=10,y=109)

        NV_True = tk.IntVar()
        NV_True.set(0)
        NV_ButtonTrue = tk.Button(cfg,text="False",bg="White",
                              command=lambda:[NV_True.set(1),self.button_lock(NV_ButtonTrue,lock=True),self.button_lock(NV_ButtonFalse,lock=False)],
                              width=4,height=1)
        NV_ButtonTrue.place(x=120,y=107)
        
        NV_ButtonFalse = tk.Button(cfg,text="True",bg="White",
                              command=lambda:[NV_True.set(0),self.button_lock(NV_ButtonFalse,lock=True),self.button_lock(NV_ButtonTrue,lock=False)],
                              width=4,height=1)
        NV_ButtonFalse.place(x=80,y=107)

        #プレイ回数
        if is_evaluate == 1:
            numlabel = tk.Label(cfg,text="Episodes")
            numlabel.place(x=10,y=134)
            numtxt = tk.Text(cfg,width=10,height=1)
            numtxt.insert(1.0,"50")
            numtxt.place(x=70,y=135)
      
            opBtn = tk.Button(cfg,text="OK",command = lambda:self.cfg_decision(comb1=combo1.get(),comb2=combo2.get(),play_num=numtxt.get("1.0","end"),map_size=MAP_combo.get(),prey_level=prey_combo.get(),non_vision=NV_True.get()))
        else:
            opBtn = tk.Button(cfg,text="OK",command = lambda:self.cfg_decision(comb1=combo1.get(),comb2=combo2.get(),map_size=MAP_combo.get(),prey_level=prey_combo.get(),non_vision=NV_True.get()))
        opBtn.place(x=260,y=135)
    
    def button_lock(self,button,lock):
        if lock == True:
            button['state'] = tk.DISABLED
        else:
            button['state'] = tk.NORMAL
    
    def cfg_decision(self,comb1,comb2,play_num=1,map_size="5x5",prey_level="Random",non_vision=0):
        #aiの選択
        if comb1 == "NashQ-Learning":
            pass
        elif comb1 == "Human":
            self.agent1 = ha.HumanAgent(0,self.actions)
            self.HAF[0] = 1
        elif comb1 == "Q-Learning":
            self.agent1 = qla.QlearningAgent(aid=0,
                                             policy=qla.EpsGreedyQPolicy(epsilon=1.,decay_rate=.99),
                                             actions=self.actions,training=False,agent_num=0,
                                             map_size=map_size,non_vision=non_vision)
        elif comb1 == "Q-Lambda":
            self.agent1 = ps.QlearningAgent(aid=0,
                                            policy=ps.EpsGreedyQPolicy(epsilon=1.,decay_rate=.99),
                                            actions=self.actions,training=False,agent_num=0,
                                            map_size=map_size,mode=0,non_vision=non_vision)
        elif comb1 == "Profit Sharing":
            self.agent1 = ps.QlearningAgent(aid=0,
                                            policy=ps.BoltzmannQPolicy(training=0),
                                            actions=self.actions,training=False,agent_num=0,
                                            map_size=map_size,mode=1,non_vision=non_vision)
        
        elif comb1 == "Improved Profit Sharing":
            self.agent1 = psm.QlearningAgent(aid=0,
                                             policy=psm.BoltzmannQPolicy(training=0),
                                             actions=self.actions,training=False,agent_num=0,
                                             map_size=map_size,mode=1,non_vision=non_vision)
            
        #label更新
        self.agent1mode.set("Agent1 : "+ comb1)

        if comb2 == "NashQ-Learning":
            pass
        elif comb2 == "Human":
            self.agent2 = ha.HumanAgent(1,self.actions)
            self.HAF[1] = 1
        elif comb2 == "Q-Learning":
            self.agent2 = qla.QlearningAgent(aid=1,
                                             policy=qla.EpsGreedyQPolicy(epsilon=1.,decay_rate=.99),
                                             actions=self.actions,training=False,agent_num=1,
                                             map_size=map_size,non_vision=non_vision)
        elif comb2 == "Q-Lambda":
            self.agent2 = ps.QlearningAgent(aid=1,
                                            policy=ps.EpsGreedyQPolicy(epsilon=1.,decay_rate=.99),
                                            actions=self.actions,training=False,agent_num=1,
                                            map_size=map_size,mode=0,non_vision=non_vision)
        elif comb2 == "Profit Sharing":
            self.agent2 = ps.QlearningAgent(aid=1,
                                            policy=ps.BoltzmannQPolicy(training=0),
                                            actions=self.actions,training=False,agent_num=1,
                                            map_size=map_size,mode=1,non_vision=non_vision)
        elif comb2 == "Improved Profit Sharing":
            self.agent2 = psm.QlearningAgent(aid=1,
                                            policy=psm.BoltzmannQPolicy(training=0),
                                            actions=self.actions,training=False,agent_num=1,
                                            map_size=map_size,mode=1,non_vision=non_vision)
            
        #label更新
        self.agent2mode.set("Agent2 : "+comb2)

        self.agents = [self.agent1,self.agent2]

        #視野制限
        self.non_vision = non_vision

        #プレイ回数
        self.play_num = play_num

        #MAPサイズ
        self.map_size = map_size

        #敵のレベル
        self.prey_level = prey_level
        self.prey_mode.set("Prey : "+prey_level)
    
    def set_game(self,map_size):
        #tag付け
        if map_size == "5x5":
            for i,x in zip("12345",range(50,300,50)):
                for j,y in zip("abcde",range(50,300,50)):
                    #pos: 長方形の４頂点
                    pos = x, y, x+50, y+50
                    tag = j + i
                    #tagの位置
                    self.tag2pos[tag] = pos
                    #その位置を長方形で塗りつぶし
                    self.cv.create_rectangle(*pos, fill="white",tags=tag)
                    self.z2tag[self.z_coordinate(tag)] = tag
                    #cv.tag_bind(tag,"<ButtonPress-1>",pressed)
        
        elif map_size == "3x3":
            for i,x in zip("123",range(100,250,50)):
                for j,y in zip("abc",range(100,250,50)):
                    #pos: 長方形の４頂点
                    pos = x, y, x+50, y+50
                    tag = j + i
                    #tagの位置
                    self.tag2pos[tag] = pos
                    #その位置を長方形で塗りつぶし
                    self.cv.create_rectangle(*pos, fill="white",tags=tag)
                    self.z2tag[self.z_coordinate(tag)] = tag
                    #cv.tag_bind(tag,"<ButtonPress-1>",pressed)

        elif map_size == "7x7":
            for i,x in zip("1234567",range(0,350,50)):
                for j,y in zip("abcdefg",range(0,350,50)):
                    #pos: 長方形の４頂点
                    pos = x, y, x+50, y+50
                    tag = j + i
                    #tagの位置
                    self.tag2pos[tag] = pos
                    #その位置を長方形で塗りつぶし
                    self.cv.create_rectangle(*pos, fill="white",tags=tag)
                    self.z2tag[self.z_coordinate(tag)] = tag
                    #cv.tag_bind(tag,"<ButtonPress-1>",pressed)

        
    def reset(self,mode=0):
        #盤面の初期化
        self.cv.delete("all")
        if mode == 1:
            self.thread_stop = 1
            self.StartBtn['state'] = tk.NORMAL
    
    def thread(self,is_evaluate=0,non_vision=0):
        #thread処理(tkinterを動かしながら他のコードを動かすのに必須)
        self.thread_stop = 0
        self.StartBtn['state'] = tk.DISABLED
        th_gameloop = threading.Thread(target=self.gameloop,args=(is_evaluate,non_vision))
        th_gameloop.start()
    
    def load_q_table(self):
        pass
    
    def update(self,board,map_size="5x5"):
        #図形が増え続けるので削除
        self.cv.delete("agents","prey")

        #現在のmap取得
        if map_size == "5x5":
            rb = board.reshape(25,1)
            for i,n in zip("abcde",(0,5,10,15,20)):
                for j in ("12345"):
                    t = i + j
                    x = n + int(j) - 1
                    if "A" == rb[x]:
                        #agentならred
                        self.cv.create_oval(*self.tag2pos[t],fill="red",tag="agents")
                    if "P" == rb[x]:
                        #preyならblue
                        self.cv.create_oval(*self.tag2pos[t],fill="blue",tag="prey")

        elif map_size == "3x3":
            rb = board.reshape(9,1)
            for i,n in zip("abc",(0,3,6)):
                for j in ("123"):
                    t = i + j
                    x = n + int(j) - 1
                    if "A" == rb[x]:
                        #agentならred
                        self.cv.create_oval(*self.tag2pos[t],fill="red",tag="agents")
                    if "P" == rb[x]:
                        #preyならblue
                        self.cv.create_oval(*self.tag2pos[t],fill="blue",tag="prey")

        elif map_size == "7x7":
            rb = board.reshape(49,1)
            for i,n in zip("abcdefg",(0,7,14,21,28,35,42)):
                for j in ("1234567"):
                    t = i + j
                    x = n + int(j) - 1
                    if "A" == rb[x]:
                        #agentならred
                        self.cv.create_oval(*self.tag2pos[t],fill="red",tag="agents")
                    if "P" == rb[x]:
                        #preyならblue
                        self.cv.create_oval(*self.tag2pos[t],fill="blue",tag="prey")

        
    def z_coordinate(self,tag):
        x = "1234567".index(tag[1])+1
        y = "abcdefg".index(tag[0])+1
        return y*10 + x

    def gameloop(self,is_evaluate=0,non_vision=0):
        #ゲームの実行
        if self.agents == []:
            messagebox.showerror("エラー","agentの種類を選択して下さい")
            return 0
        
        try:
            self.play_num = int(self.play_num)
        except:
            messagebox.showerror("エラー","episodesには数字を入力してください")
        
        #MAPサイズ
        map_size = self.map_size

        #敵のレベル
        prey_level = self.prey_level

        #ゲーム
        game = pp.PredatorsPursuitGame(self.agents,is_evaluate,map_size,prey_level)

        result_coop = 0

        for _num in range(self.play_num):

            #初期設定
            is_capture = False
            agents_pos = game.get_agents_pos()
            print(agents_pos)
            board = game.print_current_map()
            self.reset()
            self.set_game(map_size)
            self.update(board,map_size)

            #協調行動となるstep
            cooperative_step = 0

            #初期位置の取得
            for agent in self.agents:
                agent.observe(agents_pos[agent.aid])

            #step数
            nb_step = 1

            while is_capture is False:
                #スレッドを停止するかどうか
                if self.thread_stop == 1:
                    sys.exit()

                actions = []

                #行動選択
                for n,agent in enumerate(self.agents):
                    #playerがHumanかどうか判定
                    if self.HAF[n] == 0:
                        action = agent.act()
                    else:
                        #ボタン押せるようにする関数
                        self.action_button_able()
                        #疑似的入力待ち
                        while(self.var.get() == 5):
                            pass
                            #print("入力待ち")
                        action = self.var.get()
                        self.var.set(5)
                        #ボタンを押せなくする関数
                        self.action_button_disable()

                    while game.in_map(agent.aid,action) is False:
                        if self.HAF[n] == 0:
                            action = agent.react()
                        else:
                            #ボタン押せるようにする関数
                            self.action_button_able()
                            #疑似的入力待ち
                            while(self.var.get() == 5):
                                pass
                                #print("入力待ち")
                            action = self.var.get()
                            self.var.set(5)
                            #ボタンを押せなくする関数
                            self.action_button_disable()
                            
                    actions.append(action)

                    #テキスト追加
                    print("agent:",agent.aid,"choice:",action)
                    straid = str(agent.aid)
                    stract = str(action)
                    strstep = str(nb_step)
                    self.movelog.insert('end',"\nstep:"+strstep+"\nagent:"+straid+"\nchoice:"+stract)
                    self.scrollbar.pack(side=tk.RIGHT,fill="y")
                
                #学習
                states,r1,r2,is_capture = game.step(actions,nb_step)

                #視野制限の場合
                if non_vision == 1:
                    rng = [-2,-1,0,1,2]
                    mod_states = {0:[(100,100),(100,100),(100,100)],1:[(100,100),(100,100),(100,100)]}

                    for i,_obj in enumerate(self.agents):
                        px,py = states[i][0]
                        mod_states[i][0] = states[i][0]

                        for x,y in itertools.product(rng,rng):
                            #周囲探索
                            to_py = py + y
                            to_px = px + x
                            to_pr = (to_px,to_py)

                            if ((to_pr) == states[i][1]):
                                #agentがいたら
                                mod_states[i][1] = states[i][1]
                            
                            if ((to_pr) == states[i][2]):
                                #Preyがいたら
                                mod_states[i][2] = states[i][2]
                
                    states = mod_states

                self.agent1.observe(states[0],reward=r1)
                self.agent2.observe(states[1],reward=r2)

                #map更新
                board = game.print_current_map()
                self.update(board,map_size)
                print(board)

                #協調行動の評価
                if is_evaluate == 1:
                    #協調行動カウントしたい
                    is_cooperate_state = 0

                    #agent同士の位置の差
                    x_pos_dif, y_pos_dif = game.position_difference()

                    #協調状態かどうか判定
                    if (x_pos_dif, y_pos_dif) == (1,1):
                        #差が1,1なら
                        is_cooperate_state += game.search_1_1()

                    elif (x_pos_dif, y_pos_dif) == (2,0) or (x_pos_dif, y_pos_dif) == (0,2):
                        #差が2,0か0,2なら
                        is_cooperate_state += game.search_2_0()
                    
                    elif (x_pos_dif, y_pos_dif) == (2,2):
                        #差が2,2なら
                        is_cooperate_state += game.search_2_0()
                        is_cooperate_state += game.search_2_1()
                    
                    elif (x_pos_dif, y_pos_dif) == (1,0) or (x_pos_dif, y_pos_dif) == (0,1):
                        #差が1,0か0,1なら
                        is_cooperate_state += game.search_1_0()
                    
                    elif (x_pos_dif,y_pos_dif) == (2,1) or (x_pos_dif,y_pos_dif) == (1,2):
                        #差が2,1か1,2なら
                        is_cooperate_state += game.search_2_1()

                    else:
                        is_cooperate_state = 0

                    if is_cooperate_state >= 1:
                        cooperative_step += 1
                    
                print("step:",nb_step)
                print("cooperative_step: ",cooperative_step)
                print("-----------------------------------")
                time.sleep(0.05)
                if is_capture == False:
                    nb_step += 1

            if is_evaluate == 1:
                try:
                    p = (cooperative_step / nb_step)*100
                except:
                    p = 0
                
                #messagebox.showinfo("結果",res)
                result_coop += p

            game.reset()

        if is_evaluate == 1:
            try:
                result_coop = result_coop / self.play_num
            except:
                result_coop = 0

            res = "Cooperative actions rate：" + str(f'{result_coop:.02f}') +"%"
            messagebox.showinfo("result",res)
        
        self.StartBtn['state'] = tk.NORMAL

    #-------------------------------------------------------------------
    #学習mode

    def make_rlmode(self):
        #学習用UI
        self.rl = tk.Toplevel()
        self.rl.title("Learning")
        self.rl.geometry("300x280")

        #学習回数の設定
        self.RLTimesLabel = tk.Label(self.rl,text="Iteration",width=10,height=1)
        self.RLTimesLabel.place(x=0,y=3)
        self.RLTimesInput = tk.Text(self.rl,width=10,height=1)
        self.RLTimesInput.insert(1.0,"100")
        self.RLTimesInput.place(x=80,y=5)

        #学習結果をグラフ化するかどうか
        self.GraphTrue = tk.IntVar()    # 1:True,0:False
        self.GraphTrue.set(0)
        self.RLGraphLabel = tk.Label(self.rl,text="Graph",width=10,height=1)
        self.RLGraphLabel.place(x=0,y=27)
        self.RLGraphTrue = tk.Button(self.rl,text="True",bg="white",
                                     command=lambda:[self.GraphTrue.set(1),self.button_lock(self.RLGraphTrue,lock=True),self.button_lock(self.RLGraphFalse,lock=False)],
                                     width=6,height=1)
        self.RLGraphTrue.place(x=80,y=25)
        self.RLGraphFalse = tk.Button(self.rl,text="False",bg="white",
                                      command=lambda:[self.GraphTrue.set(0),self.button_lock(self.RLGraphFalse,lock=True),self.button_lock(self.RLGraphTrue,lock=False)],
                                      width=6,height=1)
        self.RLGraphFalse.place(x=160,y=25)

        #学習方法の選択
        self.RLMethodLabel = tk.Label(self.rl,text="Method1",width=10,height=1)
        self.RLMethodLabel.place(x=0,y=55)
        self.RLMethodCombo = ttk.Combobox(self.rl,state="readonly")
        self.RLMethodCombo["values"] = ("Q-Learning","Q-Lambda","Profit Sharing","Improved Profit Sharing")
        self.RLMethodCombo.current(0)
        self.RLMethodCombo.place(x=80,y=55)

        self.RLMethodLabel2 = tk.Label(self.rl,text="Method2",width=10,height=1)
        self.RLMethodLabel2.place(x=0,y=82)
        self.RLMethodCombo2 = ttk.Combobox(self.rl,state="readonly")
        self.RLMethodCombo2["values"] = ("None","Q-Learning","Q-Lambda","Profit Sharing","Improved Profit Sharing")
        self.RLMethodCombo2.current(0)
        self.RLMethodCombo2.place(x=80,y=82)

        self.RLMethodLabel3 = tk.Label(self.rl,text="Method3",width=10,height=1)
        self.RLMethodLabel3.place(x=0,y=109)
        self.RLMethodCombo3 = ttk.Combobox(self.rl,state="readonly")
        self.RLMethodCombo3["values"] = ("None","Q-Learning","Q-Lambda","Profit Sharing","Improved Profit Sharing")
        self.RLMethodCombo3.current(0)
        self.RLMethodCombo3.place(x=80,y=109)

        #Agentsの数の選択(未実装) → ２のみ
        self.AgentsNumLabel = tk.Label(self.rl,text="Agents num",width=10,height=1)
        self.AgentsNumLabel.place(x=0,y=137)
        self.AgentsNumCombo = ttk.Combobox(self.rl,state="readonly",width=10,height=1)
        self.AgentsNumCombo["value"] = ("2","3","4")
        self.AgentsNumCombo.current(0)
        self.AgentsNumCombo.place(x=80,y=137)

        #MAPの選択
        self.MAPNumLabel = tk.Label(self.rl,text="Map size",width=10,height=1)
        self.MAPNumLabel.place(x=0,y=164)
        self.MAPNumCombo= ttk.Combobox(self.rl,state="readonly",width=10,height=1)
        self.MAPNumCombo["value"] = ("3x3","5x5","7x7","7x7 non-vision")
        self.MAPNumCombo.current(0)
        self.MAPNumCombo.place(x=80,y=164)

        #ターゲットの逃げ方
        self.TargetRunLabel = tk.Label(self.rl,text="Target",width=10,height=1)
        self.TargetRunLabel.place(x=0,y=191)
        self.TargetRunCombo = ttk.Combobox(self.rl,state="readonly",width=10,height=1)
        self.TargetRunCombo["value"] = ("Random","1-Square-Search","No-Move")
        self.TargetRunCombo.current(0)
        self.TargetRunCombo.place(x=80,y=191)

        """
        #視野制限
        self.VisibilityLabel = tk.Label(self.rl,text="Visibility",width=10,height=1)
        self.VisibilityLabel.place(x=0,y=218)
        self.Visible_True = tk.IntVar()
        self.Visible_True.set(0)
        self.VisibleTrue = tk.Button(self.rl,text="True",bg="White",
                              command=lambda:[self.Visible_True.set(0),self.button_lock(self.VisibleTrue,lock=True),self.button_lock(self.VisibleFalse,lock=False)],
                              width=6,height=1)
        self.VisibleTrue.place(x=80,y=218)
        self.VisibleFalse = tk.Button(self.rl,text="False",bg="White",
                              command=lambda:[self.Visible_True.set(1),self.button_lock(self.VisibleFalse,lock=True),self.button_lock(self.VisibleTrue,lock=False)],
                              width=6,height=1)
        self.VisibleFalse.place(x=160,y=218)
        """

        #状態行動価値の保存
        self.SaveQLabel = tk.Label(self.rl,text="Save value",width=10,height=1)
        self.SaveQLabel.place(x=0,y=218)

        self.SaveTrue = tk.IntVar()    # 1:True,0:False
        self.SaveTrue.set(0)
        self.SaveQTrue = tk.Button(self.rl,text="True",bg="white",
                                     command=lambda:[self.SaveTrue.set(1),self.button_lock(self.SaveQTrue,lock=True),self.button_lock(self.SaveQFalse,lock=False)],
                                     width=6,height=1)
        self.SaveQTrue.place(x=80,y=218)
        self.SaveQFalse = tk.Button(self.rl,text="False",bg="white",
                                     command=lambda:[self.SaveTrue.set(0),self.button_lock(self.SaveQFalse,lock=True),self.button_lock(self.SaveQTrue,lock=False)],
                                     width=6,height=1)
        self.SaveQFalse.place(x=160,y=218)

        #学習の実行
        self.TrainStartButton = tk.Button(self.rl,text="START",
                                    command=lambda: self.train_start(),
                                    width=7,height=1)
        self.TrainStartButton.place(x=230,y=245)
    

    def progress_msg(self,train_num):
        #progress messageの表示
        self.popup = tk.Toplevel()
        self.popup.title("Learning Progress")
        self.popup.geometry("150x50")
        self.label = tk.Label(self.popup,text="Learning...")
        self.progress = ttk.Progressbar(self.popup,orient=tk.HORIZONTAL,value=self.episode,
                                        length=100,maximum=train_num,mode='determinate')
        self.label.pack()
        self.progress.pack()

    def train_start(self):
        self.agents = []
        self.actions = np.arange(5)

        #入力した値を取得していく
        #学習回数
        train_num_str = self.RLTimesInput.get("1.0","end")
        
        #文字列が数字かどうか判定
        try:
            train_num = int(train_num_str)
        except :
            messagebox.showerror("エラー","学習回数には数字を入力してください")
            return 0
        
        #グラフ化するかどうか
        graph_flag = self.GraphTrue.get()

        #マップサイズ
        non_vision = 0
        if self.MAPNumCombo.get() == "7x7 non-vision":
            map_size = "7x7"
            non_vision = 1
        elif self.MAPNumCombo.get() == "7x7":
            map_size = "7x7"
        elif self.MAPNumCombo.get() == "5x5":
            map_size = "5x5"
        elif self.MAPNumCombo.get() == "3x3":
            map_size = "3x3"
        else:
            messagebox.showerror("エラー","正しいマップサイズを入力してください")
        
        #Q値を保存するかどうか
        save_flag = self.SaveTrue.get()

        #学習方法
        agents_temp = []
        if self.RLMethodCombo.get() == "Q-Learning":
            for i in range(int(self.AgentsNumCombo.get())):
                temp = qla.QlearningAgent(aid=i,
                                          policy=qla.EpsGreedyQPolicy(epsilon=1.,decay_rate=.99),
                                          actions=self.actions,training=True,agent_num=i,
                                          map_size=map_size,non_vision=non_vision)
                agents_temp.append(temp)
        
        elif self.RLMethodCombo.get() == "Q-Lambda":
            for i in range(int(self.AgentsNumCombo.get())):
                temp = ps.QlearningAgent(aid=i,
                                         policy=ps.EpsGreedyQPolicy(epsilon=1.,decay_rate=.99),
                                         actions=self.actions,training=True,agent_num=i,mode=0,
                                         map_size=map_size,non_vision=non_vision)
                agents_temp.append(temp)
        
        elif self.RLMethodCombo.get() == "Profit Sharing":
            for i in range(int(self.AgentsNumCombo.get())):
                temp = ps.QlearningAgent(aid=i,
                                         policy=ps.BoltzmannQPolicy(training=1),
                                         actions=self.actions,training=True,agent_num=i,mode=1,
                                         map_size=map_size,non_vision=non_vision)
                agents_temp.append(temp)
        
        elif self.RLMethodCombo.get() == "Improved Profit Sharing":
            for i in range(int(self.AgentsNumCombo.get())):
                temp = psm.QlearningAgent(aid=i,
                                         policy=psm.BoltzmannQPolicy(training=1),
                                         actions=self.actions,training=True,agent_num=i,mode=1,
                                         map_size=map_size,non_vision=non_vision)
                agents_temp.append(temp)
        
        elif self.RLMethodCombo.get() == "Nash Q-Learning":
            messagebox.showerror("エラー","未実装です。")
            return 0

        elif self.RLMethodCombo.get() == "None":
            pass
        
        self.agents.append(agents_temp)

        #学習方法2
        agents_temp = []
        if self.RLMethodCombo2.get() == "Q-Learning":
            for i in range(int(self.AgentsNumCombo.get())):
                temp = qla.QlearningAgent(aid=i,
                                          policy=qla.EpsGreedyQPolicy(epsilon=1.,decay_rate=.99),
                                          actions=self.actions,training=True,agent_num=i,
                                          map_size=map_size,non_vision=non_vision)
                agents_temp.append(temp)
        
        elif self.RLMethodCombo2.get() == "Q-Lambda":
            for i in range(int(self.AgentsNumCombo.get())):
                temp = ps.QlearningAgent(aid=i,
                                         policy=ps.EpsGreedyQPolicy(epsilon=1.,decay_rate=.99),
                                         actions=self.actions,training=True,agent_num=i,mode=0,
                                         map_size=map_size,non_vision=non_vision)
                agents_temp.append(temp)
        
        elif self.RLMethodCombo2.get() == "Profit Sharing":
            for i in range(int(self.AgentsNumCombo.get())):
                temp = ps.QlearningAgent(aid=i,
                                         policy=ps.BoltzmannQPolicy(training=1),
                                         actions=self.actions,training=True,agent_num=i,mode=1,
                                         map_size=map_size,non_vision=non_vision)
                agents_temp.append(temp)
        
        elif self.RLMethodCombo2.get() == "Improved Profit Sharing":
            for i in range(int(self.AgentsNumCombo.get())):
                temp = psm.QlearningAgent(aid=i,
                                         policy=psm.BoltzmannQPolicy(training=1),
                                         actions=self.actions,training=True,agent_num=i,mode=1,
                                         map_size=map_size,non_vision=non_vision)
                agents_temp.append(temp)
        
        elif self.RLMethodCombo2.get() == "Nash Q-Learning":
            messagebox.showerror("エラー","未実装です。")
            return 0

        elif self.RLMethodCombo2.get() == "None":
            pass
        
        self.agents.append(agents_temp)

        #学習方法3
        agents_temp = []
        if self.RLMethodCombo3.get() == "Q-Learning":
            for i in range(int(self.AgentsNumCombo.get())):
                temp = qla.QlearningAgent(aid=i,
                                          policy=qla.EpsGreedyQPolicy(epsilon=1.,decay_rate=.99),
                                          actions=self.actions,training=True,agent_num=i,
                                          map_size=map_size,non_vision=non_vision)
                agents_temp.append(temp)
        
        elif self.RLMethodCombo3.get() == "Q-Lambda":
            for i in range(int(self.AgentsNumCombo.get())):
                temp = ps.QlearningAgent(aid=i,
                                         policy=ps.EpsGreedyQPolicy(epsilon=1.,decay_rate=.99),
                                         actions=self.actions,training=True,agent_num=i,mode=0,
                                         map_size=map_size,non_vision=non_vision)
                agents_temp.append(temp)
        
        elif self.RLMethodCombo3.get() == "Profit Sharing":
            for i in range(int(self.AgentsNumCombo.get())):
                temp = ps.QlearningAgent(aid=i,
                                         policy=ps.BoltzmannQPolicy(training=1),
                                         actions=self.actions,training=True,agent_num=i,mode=1,
                                         map_size=map_size,non_vision=non_vision)
                agents_temp.append(temp)
        
        elif self.RLMethodCombo3.get() == "Improved Profit Sharing":
            for i in range(int(self.AgentsNumCombo.get())):
                temp = psm.QlearningAgent(aid=i,
                                         policy=psm.BoltzmannQPolicy(training=1),
                                         actions=self.actions,training=True,agent_num=i,mode=1,
                                         map_size=map_size,non_vision=non_vision)
                agents_temp.append(temp)
        
        elif self.RLMethodCombo3.get() == "Nash Q-Learning":
            messagebox.showerror("エラー","未実装です。")
            return 0

        elif self.RLMethodCombo3.get() == "None":
            pass
        
        self.agents.append(agents_temp)

        #空のリストの削除
        abc = [x for x in self.agents if x != [] ]
        self.agents = abc
        
        self.trainthread(graph_flag,map_size,train_num,non_vision,save_flag)

    def trainthread(self,graph_flag,map_size,train_num,non_vision,save_flag):
        #thread処理(tkinterを動かしながら他のコードを動かすのに必須)
        self.episode = 0
        self.result = []
        th_trainloop = threading.Thread(target=self.trainloop,args=[graph_flag,map_size,train_num,non_vision,save_flag])
        th_trainloop.start()
        self.progress_msg(train_num)
        while th_trainloop.is_alive()==True:
            self.progress.configure(value=self.episode)
            self.progress.update()

        self.popup.destroy()
        
        #グラフ化
        if graph_flag == 1:
            #color
            color_list = ["blue","red","Magenta"]
            #label
            label_list = []
            if self.RLMethodCombo.get() != "None":
                label_list.append(self.RLMethodCombo.get())
            if self.RLMethodCombo2.get() != "None":
                label_list.append(self.RLMethodCombo2.get())
            if self.RLMethodCombo3.get() != "None":
                label_list.append(self.RLMethodCombo3.get())
            #marker
            marker_list = ["*","^","D"]
            #graph
            fig,ax = plt.subplots()
            #軸
            ax.set_xlabel("Episodes")
            ax.set_ylabel("Steps")
            ax.set_xlim([0,train_num])
            ax.grid()

            for index in range(len(self.agents)):
                self.result[index] = pd.Series(self.result[index]).rolling(1000).mean().tolist()
                ax.plot(np.arange(len(self.result[index])),self.result[index],color=color_list[index],
                         lw=1,label=label_list[index],marker=marker_list[index],markevery=1000,
                         markersize=8)

            ax.legend(loc=0,fontsize=18)
            fig.tight_layout()
            plt.show()

        messagebox.showinfo("info","Finish")
    
    def trainloop(self,graph,map_size,train_num,non_vision,save_flag):
        rng = [-2,-1,0,1,2]
        nb_episode = train_num
        for index, _obj in enumerate(self.agents):
            agents = copy.deepcopy(self.agents[index])
            game = pp.PredatorsPursuitGame(agents=agents,map_size=map_size,prey_level=self.TargetRunCombo.get())
            res = []

            for episode in range(nb_episode):
                is_capture = False
                agents_pos = game.get_agents_pos()

                for agent in agents:
                    agent.observe(agents_pos[agent.aid],is_learn=False)

                nb_step = 1
                while is_capture is False and nb_step <= 400:
                    actions=[]
                    for agent in agents:
                        action = agent.act()
                        while game.in_map(agent.aid,action) is False:
                            action = agent.act()
                        actions.append(action)
                    
                    states,r1,r2,is_capture = game.step(actions,nb_step)

                    #視界制限の場合
                    if non_vision == 1:
                        mod_states = {0:[(100,100),(100,100),(100,100)],1:[(100,100),(100,100),(100,100)]}
                        for i,_obj in enumerate(agents):
                            px,py = states[i][0]
                            mod_states[i][0] = states[i][0]

                            for x,y in itertools.product(rng,rng):
                                #周囲探索
                                to_py = py + y
                                to_px = px + x
                                to_pr = (to_px,to_py)

                                if ((to_pr) == states[i][1]):
                                    #agentがいたら
                                    mod_states[i][1] = states[i][1]
                                
                                if ((to_pr) == states[i][2]):
                                    #Preyがいたら
                                    mod_states[i][2] = states[i][2]
                    
                        states = mod_states

                    agents[0].observe(next_state=states[0],reward=r1,reward_o=r2,opponent_action=agents[1].previous_action,is_learn=False)
                    agents[1].observe(next_state=states[1],reward=r2,reward_o=r1,opponent_action=agents[0].previous_action,is_learn=False)
                    nb_step += 1

                res.append(nb_step)
                self.episode = episode
                game.reset()
            
            #Q値の保存
            if save_flag == 1:
                agents[0].save_q(0)
                agents[1].save_q(1)
            
            self.result.append(res)

    #------------------------------------------------------------------
    #評価モード

    def make_evmode(self):
        self.ev = tk.Toplevel()
        self.ev.title("Evaluation")
        self.ev.geometry("{}x{}".format(350,500))
        self.cv = tk.Canvas(self.ev,bg="white",width=350,height=350)
        self.cv.place(x=0,y=0)
        self.tag2pos = {}
        self.z2tag = {}

        self.agents = []
        self.actions = np.arange(5)

        self.non_vision = 0

        self.HAF = [0,0]

        self.StartBtn = tk.Button(self.ev,text="Start",command = lambda: self.thread(is_evaluate=1,non_vision=self.non_vision), width=8)
        self.StartBtn.place(x=282,y=410)

        self.ResetBtn = tk.Button(self.ev,text="Reset",command = lambda: self.reset(mode=1), width=8)
        self.ResetBtn.place(x=282,y=440)

        self.configBtn = tk.Button(self.ev,text="config",command = lambda: self.make_config(is_evaluate=1), width=8)
        self.configBtn.place(x=282,y=470)

        self.agent1mode = tk.StringVar()
        self.agent2mode = tk.StringVar()
        self.agent1mode.set("agent1 : ")
        self.agent2mode.set("agent2 : ")

        self.agent1_label = tk.Label(self.ev,textvariable = self.agent1mode)
        self.agent1_label.place(x=10,y=360)
        self.agent2_label = tk.Label(self.ev,textvariable = self.agent2mode)
        self.agent2_label.place(x=10,y=380)

        self.prey_mode = tk.StringVar()
        self.prey_mode.set("Prey : ")

        self.prey_label = tk.Label(self.ev,textvariable = self.prey_mode)
        self.prey_label.place(x=200,y=360)

        mlbl = tk.Label(self.ev,text="action")
        mlbl.place(x=12,y=405)
        self.movelog = tk.Text(self.ev,width=20,height=5)
        self.movelog.insert(1.0,"---log---")
        self.movelog.place(x=10,y=425)

        self.textFrame = tk.Frame(self.ev,width=100,height=100)
        self.textFrame.configure()
        self.textFrame.place(x=155,y=425)

        self.scrollbar = tk.Scrollbar(self.textFrame,orient=tk.VERTICAL,command=self.movelog.yview)
        self.scrollbar.pack(side=tk.RIGHT,fill="y")
        self.scrollbar.config(command=self.movelog.yview)

    #------------------------------------------------------------------
    #RUN
    
    def run(self):
        self.mainloop()

if __name__ == "__main__":
    app = PursuitGUI()
    app.run()
