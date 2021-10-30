import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from rl_utils import build_q_table,choose_action
from rl_learning import train_each_step_episode,train_each_step_episode_q_learning
from rl_utils import build_q_table
def main():
    ACTIONS = ['up', 'down', 'left', 'right']
    m = list()
    for i in range(0, 4, 1):
        for j in range(0, 12, 1):
            m.append((i, j))
    q_table = build_q_table(48, len(ACTIONS))
    q_table.index = m
    q_table_q_learning=build_q_table(48,len(ACTIONS))
    q_table_q_learning.index=m
    loss_list_q_learning=[]
    loss_list = []
    for episode in range(100):

        if episode%10==0:
            print('now is episode',episode)
            # print(q_table)
        state=(3,0)
        A=choose_action(state,q_table)
        A2=choose_action(state,q_table_q_learning)
        q_table=train_each_step_episode(0.2,A,state,q_table)
        q_table_q_learning=train_each_step_episode_q_learning(0.2,A2,state,q_table_q_learning)
        loss=q_table.sum().sum()
        loss_q=q_table_q_learning.sum().sum()
        # print(loss)
        loss_list_q_learning.append(loss_q)
        loss_list.append(loss)
    return loss_list,\
           loss_list_q_learning,q_table,q_table_q_learning

if __name__=='__main__':
    loss_sarsa,\
    loss_q,q_table,q_learing=main()
    print(q_table)
    print(q_learing)
    plt.plot(np.arange(0,100),loss_sarsa,c='b')
    plt.plot(np.arange(0,100),loss_q,c='y')
    plt.show()
