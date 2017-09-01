# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


N_EPISODES = 500
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MAX_STEPS = 500
GAMMA = 0.95
lr = 0.001



import sys
sys.path.insert(0,'..')


class DQNAgent(object):
    def __init__(self,states,actions):
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.lr = lr
        self.memory = Memory()
        self.model = self.build_model(states,actions)


    def build_model(self,states,actions):
        model = Sequential()
        model.add(Dense(24,input_dim = states,activation = "relu"))
        model.add(Dense(24,activation = "relu"))
        model.add(Dense(actions,activation = "linear"))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.lr))
        return model


    def train_on_batch(self,batch_size = 32):
        if len(self.memory.cache) > batch_size:
            batch = random.sample(self.memory.cache, batch_size)
        else:
            batch = self.memory.cache

        for state,action,reward,next_state,done in batch:
            state = self.expand_state_vector(state)
            next_state = self.expand_state_vector(next_state)


            targets = self.model.predict(state)

            if not done:
                target = reward + GAMMA * np.amax(self.model.predict(next_state)[0])
            else:
                target = reward

            targets[0][action] = target

            self.model.fit(state,targets,epochs = 1,verbose = 0)


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def expand_state_vector(self,state):
        if len(state.shape) == 1:
            return np.expand_dims(state,axis = 0)
        else:
            return state




    def act(self,state):
        state = self.expand_state_vector(state)

        q = self.model.predict(state)

        if np.random.rand() > self.epsilon:
            a = np.argmax(q[0])
        else:
            a = np.random.randint(env.action_space.n)

        return a 



    def remember(self,state,action,reward,next_state,done):
        self.memory.save(state,action,reward,next_state,done)






class Memory(object):
    def __init__(self):
        self.cache = deque(maxlen=2000)
    
    def save(self,state,action,reward,next_state,done):
        self.cache.append((state,action,reward,next_state,done))

    def empty_cache(self):
        self.__init__()















if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    agent = DQNAgent(len(env.observation_space.high),env.action_space.n)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    EPISODES = 500
    rewards = []

    # # ITERATION OVER EPISODES
    for i_episode in range(N_EPISODES):

        s = env.reset()
        episode_reward = 0


        # EPISODE LOOP
        for i_step in range(MAX_STEPS):
        

            a = agent.act(s)
            
            # Take the action, and get the reward from environment
            s_next,r,done,info = env.step(a)

            # Tweaking the reward
            r = r if not done else -10

            # Caching to train later
            agent.remember(s,a,r,s_next,done)
                
            # Go to the next state
            s = s_next
            
            # If the episode is terminated
            if done:
                print("Episode {}/{} finished after {} timesteps - epsilon : {:.2}".format(i_episode+1,N_EPISODES,i_step,agent.epsilon))
                break
    

        # Training
        agent.train_on_batch()



    average_running_rewards = np.cumsum(rewards)/np.array(range(1,len(rewards)+1))
    plt.figure(figsize = (15,4))
    plt.plot(average_running_rewards)
    plt.show()





