import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import Q_Learning_Agent as qla
import pursuit_problem as pp
import csv

actions = np.arange(5)
agent1 = qla.QlearningAgent(aid=0,policy=qla.EpsGreedyQPolicy(epsilon=1.,decay_rate=.99),actions=actions,training=False,agent_num=0)
agent2 = qla.QlearningAgent(aid=1,policy=qla.EpsGreedyQPolicy(epsilon=1.,decay_rate=.99),actions=actions,training=False,agent_num=1)
agents = [agent1,agent2]

game = pp.PredatorsPursuitGame(agents)

result = []

is_capture = False
agents_pos = game.get_agents_pos()
num = 0

for agent in agents:
    agent.observe(agents_pos[agent.aid])

game.print_current_map()

nb_step = 1
while is_capture is False:
    num += 1
    actions = []

    for agent in agents:
        action = agent.act()
        while game.in_map(agent.aid, action) is False:
            action = agent.react()
        print("agent",agent.aid,"choice : ",action)
        actions.append(action)

    #game.print_current_map()
    states, r1,r2, is_capture = game.step(actions,nb_step)
    agent1.observe(states[0],reward=r1)
    agent2.observe(states[1],reward=r2)
    game.print_current_map()

    print("step : ",num)
    nb_step += 1

game.reset()

