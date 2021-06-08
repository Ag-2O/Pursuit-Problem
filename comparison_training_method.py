import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import itertools
import csv

import Q_Learning_Agent as qla
import Profit_Sharing as ps
import PS_mod as psm
import pursuit_problem as pp

#比較用
def gameloop(agents=None,nb_episode=10000,game=None,non_vision=0,mode=0):
    #視野
    rng = [-2,-1,0,1,2]

    #結果リスト
    result = []

    #sub_goal
    is_discover = [0,0]
    is_pray = [0,0]

    #ゲームループ
    for episode in range(nb_episode):
        is_capture = False
        agents_pos = game.get_agents_pos()

        for agent in agents:
            agent.observe(agents_pos[agent.aid])
        
        #current_map = game.print_current_map()

        nb_step = 1
        while is_capture is False:
            actions = []
            for agent in agents:
                action = agent.act()
                while game.in_map(agent.aid, action) is False:
                    action = agent.act()
                actions.append(action)
                
            states, r1, r2, is_capture = game.step(actions,nb_step)
      
            mod_states = {0:[(100,100),(100,100),(100,100)],1:[(100,100),(100,100),(100,100)]}

            #視界制限の場合
            if non_vision == 1:
                for i,_obj in enumerate(agents):
                    is_pray[i] = 0

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
                            #発見報酬
                            if mode == 5:
                                is_pray[i] = 1
                                is_discover[i] = 1

                                if i==0:
                                    r1 = r1 + 1
                                else:
                                    r2 = r2 + 1        

                    
                    if is_pray[i] == 0 and is_discover[i]==1:
                        is_discover[i] = 0      

            
                states = mod_states

            agents[0].observe(next_state=states[0],reward=r1,reward_o=r2,opponent_action=actions[1])
            agents[1].observe(next_state=states[1],reward=r2,reward_o=r1,opponent_action=actions[0])
            #current_map = game.print_current_map() #現状のマップの表示
            #print(current_map)
            nb_step += 1
            #print("step: ",nb_step)
        print("phase:",mode,"episodes",episode)
        result.append(nb_step)
        game.reset()
    #agent1.print_q()

    #q-valueの保存
    #agents[0].save_q(0)
    #agents[1].save_q(1)
    #捉えるのにかかったステップ数の移動平均の計算
    result = pd.Series(result).rolling(1000).mean().tolist()
    return result

if __name__ == '__main__':
    #学習回数
    nb_episode = 20000

    #マップサイズ
    map_size = "7x7"

    #視界制限
    non_vision = 1

    #行動
    actions = np.arange(5)

    #敵の行動
    prey_level="1-Square-Search"

    result1 = []
    result2 = []
    result3 = []
    result4 = [] 

    fig,ax = plt.subplots()

    #color
    c1,c2,c3,c4 = "blue","green","red","Magenta"

    #label
    l1,l2,l3,l4 = "q-learning","q-lambda","profit sharing","improved profit sharing"

    #軸
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Steps")
    ax.set_xlim([0,nb_episode])
    #ax.set_ylim()
    ax.grid()
    
    
    #Q-Learning
    agent1 = qla.QlearningAgent(aid=0,policy=qla.EpsGreedyQPolicy(epsilon=1.,decay_rate=.9),actions=actions,training=True,agent_num=0,map_size=map_size,non_vision=non_vision)
    agent2 = qla.QlearningAgent(aid=1,policy=qla.EpsGreedyQPolicy(epsilon=1.,decay_rate=.9),actions=actions,training=True,agent_num=1,map_size=map_size,non_vision=non_vision)
    agents1 = [agent1,agent2]
    game1 = pp.PredatorsPursuitGame(agents=agents1,map_size=map_size,prey_level=prey_level)
    result1 = gameloop(agents=agents1,nb_episode=nb_episode,game=game1,non_vision=non_vision,mode=1)
    ax.plot(np.arange(len(result1)),result1,color=c1,lw=0.5,label=l1,marker="*",markevery=1000,markersize=6)
    del game1,result1,agent1,agent2,agents1
    
    
    """
    #Q-lambda
    agent3 = ps.QlearningAgent(aid=0,policy=ps.EpsGreedyQPolicy(epsilon=1.,decay_rate=.9),actions=actions,training=True,agent_num=0,map_size=map_size,mode=0,non_vision=non_vision)
    agent4 = ps.QlearningAgent(aid=1,policy=ps.EpsGreedyQPolicy(epsilon=1.,decay_rate=.9),actions=actions,training=True,agent_num=1,map_size=map_size,mode=0,non_vision=non_vision)
    agents2 = [agent3,agent4]
    game2 = pp.PredatorsPursuitGame(agents=agents2,map_size=map_size,prey_level=prey_level)
    result2 = gameloop(agents=agents2,nb_episode=nb_episode,game=game2,non_vision=non_vision,mode=2)
    ax.plot(np.arange(len(result2)),result2,color=c2,lw=0.5,label=l2)
    result2 = []
    del game2,result2,agent3,agent4,agents2
    """
    
    #profit sharing
    agent5 = ps.QlearningAgent(aid=0,policy=ps.BoltzmannQPolicy(training=1),actions=actions,training=True,agent_num=0,map_size=map_size,mode=1,non_vision=non_vision)
    agent6 = ps.QlearningAgent(aid=1,policy=ps.BoltzmannQPolicy(training=1),actions=actions,training=True,agent_num=1,map_size=map_size,mode=1,non_vision=non_vision)
    agents3 = [agent5,agent6]
    game3 = pp.PredatorsPursuitGame(agents=agents3,map_size=map_size,prey_level=prey_level)
    result3 = gameloop(agents=agents3,nb_episode=nb_episode,game=game3,non_vision=non_vision,mode=3)
    ax.plot(np.arange(len(result3)),result3,color=c3,lw=0.5,label=l3,marker="^",markevery=1000,markersize=6)
    result3 = []
    del game3,result3,agent5,agent6,agents3
    
    
    #modified profit sharing
    agent7 = psm.QlearningAgent(aid=0,policy=psm.BoltzmannQPolicy(training=1),actions=actions,training=True,agent_num=0,map_size=map_size,mode=1,non_vision=non_vision)
    agent8 = psm.QlearningAgent(aid=1,policy=psm.BoltzmannQPolicy(training=1),actions=actions,training=True,agent_num=1,map_size=map_size,mode=1,non_vision=non_vision)
    agents4 = [agent7,agent8]
    game4 = pp.PredatorsPursuitGame(agents=agents4,map_size=map_size,prey_level=prey_level)
    result4 = gameloop(agents=agents4,nb_episode=nb_episode,game=game4,non_vision=non_vision,mode=4)
    ax.plot(np.arange(len(result4)),result4,color=c4,lw=0.5,label=l4,marker="D",markevery=1000,markersize=6)
    del game4,result4,agent7,agent8,agents4
    
    ax.legend(loc=0,fontsize=18)
    fig.tight_layout()
    #plt.savefig("Result.jpg")
    plt.show()

    #q-valueの保存
    #agent1.save_q(0)
    #agent2.save_q(1)