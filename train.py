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


#学習回数
nb_episode = 10000

#マップサイズ
map_size = "7x7"

#視界制限
non_vision = 1

#視野
rng = [-2,-1,0,1,2]

#行動
actions = np.arange(5)

#敵の行動
prey_level="1-Square-Search"
#"1-Square-Search"

#エージェントの選択
#agent1 = qla.QlearningAgent(aid=0,policy=qla.EpsGreedyQPolicy(epsilon=1.,decay_rate=.9),actions=actions,training=True,agent_num=0,map_size=map_size,non_vision=non_vision)
#agent2 = qla.QlearningAgent(aid=1,policy=qla.EpsGreedyQPolicy(epsilon=1.,decay_rate=.9),actions=actions,training=True,agent_num=1,map_size=map_size,non_vision=non_vision)

#agent1 = ps.QlearningAgent(aid=0,policy=ps.EpsGreedyQPolicy(epsilon=1.,decay_rate=.9),actions=actions,training=True,agent_num=0,map_size=map_size,mode=0,non_vision=non_vision)
#agent2 = ps.QlearningAgent(aid=1,policy=ps.EpsGreedyQPolicy(epsilon=1.,decay_rate=.9),actions=actions,training=True,agent_num=1,map_size=map_size,mode=0,non_vision=non_vision)

#agent1 = ps.QlearningAgent(aid=0,policy=ps.BoltzmannQPolicy(training=1),actions=actions,training=True,agent_num=0,map_size=map_size,mode=0,non_vision=non_vision)
#agent2 = ps.QlearningAgent(aid=1,policy=ps.BoltzmannQPolicy(training=1),actions=actions,training=True,agent_num=1,map_size=map_size,mode=0,non_vision=non_vision)

#agent1 = ps.QlearningAgent(aid=0,policy=ps.BoltzmannQPolicy(training=1),actions=actions,training=True,agent_num=0,map_size=map_size,mode=1,non_vision=non_vision)
#agent2 = ps.QlearningAgent(aid=1,policy=ps.BoltzmannQPolicy(training=1),actions=actions,training=True,agent_num=1,map_size=map_size,mode=1,non_vision=non_vision)

agent1 = psm.QlearningAgent(aid=0,policy=psm.BoltzmannQPolicy(training=1),actions=actions,training=True,agent_num=0,map_size=map_size,mode=1,non_vision=non_vision)
agent2 = psm.QlearningAgent(aid=1,policy=psm.BoltzmannQPolicy(training=1),actions=actions,training=True,agent_num=1,map_size=map_size,mode=1,non_vision=non_vision)
agents = [agent1,agent2]

#ゲームの定義
game = pp.PredatorsPursuitGame(agents=agents,map_size=map_size,prey_level=prey_level)

#結果リスト
result = []

#ゲームループ
for episode in range(nb_episode):
    is_capture = False
    agents_pos = game.get_agents_pos()

    for agent in agents:
        agent.observe(agents_pos[agent.aid])
    
    current_map = game.print_current_map()

    nb_step = 1
    while is_capture is False:
        actions = []
        for agent in agents:
            action = agent.act()
            while game.in_map(agent.aid, action) is False:
                action = agent.act()
            actions.append(action)
            
        states, r1, r2, is_capture = game.step(actions,nb_step)
        #print("states[0]:",states[0],"states[1]:",states[1])
        #print("states[0][0]type",type(states[0][0]))
        #print("states",states)
    
        mod_states = {0:[(100,100),(100,100),(100,100)],1:[(100,100),(100,100),(100,100)]}

        #視界制限の場合
        if non_vision == 1:
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
        #print("moded states:",states)

        agent1.observe(next_state=states[0],reward=r1,reward_o=r2,opponent_action=actions[1])
        agent2.observe(next_state=states[1],reward=r2,reward_o=r1,opponent_action=actions[0])
        current_map = game.print_current_map() #現状のマップの表示
        #print(current_map)
        nb_step += 1
        #print("step: ",nb_step)
    print("episodes",episode)
    result.append(nb_step)
    game.reset()
#agent1.print_q()
#捉えるのにかかったステップ数の移動平均の計算
result = pd.Series(result).rolling(50).mean().tolist()

plt.plot(np.arange(len(result)),result,lw=0.5)
plt.xlabel("Episodes")
plt.ylabel("Steps")
plt.legend()
#plt.savefig("Result.jpg")
plt.show()

#q-valueの保存
#agent1.save_q(0)
#agent2.save_q(1)