import random
import time
import gym
import numpy as np
from collections import  deque
from tensorflow.keras import  models,layers,optimizers
class DQN(object):
    def __init__(self):
        self.step=0
        self.update_freq=200
        self.replay_size=2000
        self.replay_queue=deque(maxlen=self.replay_size)
        self.model=self.create_model()
        self.target_model=self.create_model()

    def create_model(self):
        """创建一个隐藏层为100的神经网络"""
        STATE_DIM, ACTION_DIM = 2, 3
        model = models.Sequential([
            layers.Dense(100, input_dim=STATE_DIM, activation='relu'),
            layers.Dense(ACTION_DIM, activation="linear")
        ])
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(0.001))
        return model
    def choose_action(self,s,epsilon=0.05):
        if np.random.uniform()<1-epsilon+epsilon/3:
            return np.argmax(self.model.predict(np.array([s])))
        else:
            return np.random.choice([0,1,2])
    def save_mode(self,file_path='mountain.h5'):
        print('model saved')
        self.model.save(file_path)
    def remember(self, s, a, next_s, reward):
        """历史记录，position >= 0.4时给额外的reward，快速收敛"""
        if next_s[0] >= 0.4:
            reward += 1
        self.replay_queue.append((s, a, next_s, reward))
    def train(self,batch_size=64,lr=1,factor=0.8):
        if len(self.replay_queue)<self.replay_size:
            return
        self.step+=1
        if self.step%self.update_freq==0:
            self.target_model.set_weights(self.model.get_weights())
        replay_batch=random.sample(self.replay_queue,batch_size)
        s_batch=np.array([i[0] for i in replay_batch])
        next_s_batch=np.array(i[2] for i in replay_batch)
        q=self.model.predict(s_batch)
        q_next=self.target_model.predict(next_s_batch)
        for i ,replay in enumerate(replay_batch):
            _,a,_,reward=replay
            q[i][a]=(1 - lr) * q[i][a] + lr * (reward + factor * np.amax(q_next[i]))
        self.model.fit(s_batch, q, verbose=0)