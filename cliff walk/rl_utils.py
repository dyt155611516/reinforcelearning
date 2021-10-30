import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
epsilon=0.1
def build_q_table(n_state,actons):
    ACTIONS = ['up', 'down', 'left', 'right']
    table=pd.DataFrame(np.zeros((n_state,actons)),columns=ACTIONS)
    return table
def build_env(row,col):
    env=pd.DataFrame(np.zeros((row,col)))
    return env
def choose_action(state,q_table):
    ACTIONS = ['up', 'down', 'left', 'right']
    state_actions=q_table[q_table.index==state].values
    if(np.random.uniform()<1-epsilon+epsilon/4)or(state_actions.sum()==0):
        action_name=np.random.choice(ACTIONS)
    else:
        action_name=state_actions.argmax()
        action_name=ACTIONS[action_name]
    return action_name
def get_env_feedback(state,action):
#     print('ppppppppp',state,state[0],action)
#     print(state,type(state[0]),type(state[1]))
    if (state[0]==2)&(state[1]>0)&(state[1]<11&(action=='down')):
        state_next=(3,0)
        R=-100
#         print('oooo',state_next,R)
        return state_next,R
    if (state[0]==3)&(state[1]==0)&(action=='right'):
        state_next=(3,0)
        R=-100
#         print('oooo',state_next,R)
        return state_next,R
    if action=="down":##向下
        if state==(2,11):
            state_next=(state[0]+1,state[1])
            R=100
        else:
            if state[0]==3:
                state_next=state
                R=-1
            else:
                state_next=(state[0]+1,state[1])
                R=-1
#         print('oooo',state_next,R)
        return state_next,R
    if action=="up":##向上
        if state[0]==0:
            state_next=state
            R=-1
        else:
            state_next=(state[0]-1,state[1])
            R=-1
#         print('oooo',state_next,R)
        return state_next,R
    if action=="left":##向左
        if state[1]==0:
            state_next=state
            R=-1
        else:
            state_next=(state[0],state[1]-1)
            R=-1
#         print('oooo',state_next,R)
        return state_next,R
    if action=="right":##向右
        if state==(3,0):
            state_next=state
            R=-100
            return state_next, R
        if state[1]==11:
            state_next=state
            R=-1
        else:
            state_next=(state[0],state[1]+1)
            R=-1
#         print('oooo',state_next,R)
        return state_next,R
def return_loss(q_table):
    startpoint=(3.0)
    terminalflag=True
    loss=0
    while terminalflag:
        next_ac=q_table[q_table.index==startpoint].values[0].argmax()
        if next_ac==0:
            startpoint=(startpoint[0]-1,startpoint[1])
        if next_ac==1:
            startpoint=(startpoint)