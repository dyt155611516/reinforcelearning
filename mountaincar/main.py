import time

import gym
import numpy as np
from collections import deque
from tensorflow.keras import  models,layers,optimizers
from train_util import  DQN
def train_model():
    env=gym.make('MountainCar-v0')
    episode=1000
    score_list=[]
    agent=DQN()
    for i in range(episode):
        state=env.reset()
        score=0
        while True:
            # env.render()
            a=agent.choose_action(state)
            next_state,reward,done,_=env.step(a)
            agent.remember(state,a,next_state,reward)
            agent.train(lr=0.5)
            score+=reward
            state=next_state
            if done:
                score_list.append(score)
                print('当前是episode',i,'score是',score,'最大score是',max(score_list))
                break
        if np.mean(score_list[-10:])>-160:
            agent.save_model()
            break
    env.close()
def main():
    env=gym.make('MountainCar-v0')
    train_model()
    model=models.load_model('mountain.h5')
    s=env.reset()
    score=0
    while True:
        env.render()
        time.sleep(0.01)
        a = np.argmax(model.predict(np.array([s]))[0])
        s, reward, done, _ = env.step(a)
        score += reward
        if done:
            print('score:', score)
            break
    env.close()
if __name__=='__main__':
    main()


