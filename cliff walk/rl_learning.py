import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from rl_utils import choose_action,get_env_feedback
gamma=0.2
step_counter=0
ACTIONS = ['up', 'down', 'left', 'right']
# sarsa
def train_each_step_episode(gamma,action,state,q_table):
    is_terminal = False
    while is_terminal==False:
        state_next,R=get_env_feedback(state,action)
        action_next=choose_action(state_next,q_table)
        new_q_value=q_table[q_table.index==state].loc[:,action].values+gamma*(R+0.9*q_table[q_table.index==state_next].loc[:,action_next].values-\
                                                               q_table[q_table.index==state].loc[:,action].values)
        orig_vector=q_table[q_table.index==state].values
        if action=='up':
            orig_vector[0][0]=new_q_value[0]
        if action=='down':
            orig_vector[0][1]=new_q_value[0]
        if action=='left':
            orig_vector[0][2]=new_q_value[0]
        if action=='right':
            orig_vector[0][3]=new_q_value[0]
        q_table[q_table.index==state]=orig_vector

        state=state_next
        action=action_next
        if state==(3,11):
            is_terminal=True
    # with os.open('./result/q_table.txt','a+') as w1:
    #     w1.write(q_table)
    #     w1.close()
    # print(q_table)
    return q_table
def train_each_step_episode_q_learning(gamma,action,state,q_table):
    is_terminal = False
    while is_terminal==False:
        state_next,R=get_env_feedback(state,action)
        action_next=choose_action(state_next,q_table)
        new_q_value=q_table[q_table.index==state].loc[:,action].values+gamma*(R+0.9*q_table[q_table.index==state_next].values.max()-\
                                                               q_table[q_table.index==state].loc[:,action].values)
        # print("此时",q_table[q_table.index==state].loc[:,action].values,q_table[q_table.index==state_next].values.max(),new_q_value)

        orig_vector=q_table[q_table.index==state].values
        if action=='up':
            orig_vector[0][0]=new_q_value[0]
        if action=='down':
            orig_vector[0][1]=new_q_value[0]
        if action=='left':
            orig_vector[0][2]=new_q_value[0]
        if action=='right':
            orig_vector[0][3]=new_q_value[0]
        q_table[q_table.index==state]=orig_vector
        # print('此时 状态是',state,'动作是',action,'下一个状态是',state_next,'该状态下最大的收益动作是',q_table[q_table.index==state_next].values[0].argmax(),'值是',\
        #       q_table[q_table.index==state_next].values.max(),'new q是',new_q_value,orig_vector)
        # print('111')
        state=state_next
        action=action_next
        if state==(3,11):
            is_terminal=True
    # with os.open('./result/q_table.txt','a+') as w1:
    #     w1.write(q_table)
    #     w1.close()
    # print(q_table)
    return q_table
