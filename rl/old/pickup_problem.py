# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:38:48 2016

@author: talvesdacosta
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import time

plt.style.use('ggplot')





class Point():
    def __init__(self,size=(8,8),moving = False,is_AI = False,no_positions = [],goal = [],position = None):
        '''INITIALIZATION
        A point is set on a predefined environment grid
        The position can be manually or randomly set
        One point can be either an AI or a pickup point
        '''
        self.size = size
        self.height,self.width = size
        if position == None:
            position = (random.randint(0,self.height-1),random.randint(0,self.width-1))
            while position in no_positions:
                position = (random.randint(0,self.height-1),random.randint(0,self.width-1))
        self.position = position
        self.x,self.y = self.position
        self.goal = goal
        self.moving = True if is_AI else moving 
        self.possible_moves = [(1,0),(-1,0),(0,1),(0,-1),(0,0)]
        self.value = 1 if is_AI else -1
        self.is_AI = is_AI
    
    def __repr__(self):
        '''PRINT METHOD'''
        return str(self.position)
    
    def __str__(self):
        '''PRINT METHOD'''
        return str(self.position)
    
    '''-----------------------------------------------------------------------------------------'''
    def set_position(self,new_position):
        '''To set a new position'''
        self.position = new_position
        self.x,self.y = self.position
        
    def set_goal(self,goal):
        '''To set a new goal'''
        self.goal = goal
        
    
    def move(self,action = None,impossible_positions = []):
        '''Move method, which will be called by an environment update'''
        if self.is_AI:
            '''Moves for the AI can be random or defined
            If the AI goes out of the environnement, the method returns False meaning it has lost the game. 
            If the AI reaches its goal of catching all the points, the method returns True meaning it has won the game.
            '''
            if action == None:
                default_action = random.choice(self.possible_moves)
                new_position = [sum(x) for x in zip(self.position,default_action)]
            else:
                new_position = [sum(x) for x in zip(self.position,action)]
                
            new_position = (new_position[0],new_position[1])
            self.set_position(new_position)
            if self.out_of_bounds():
                return False
            elif self.reach_goal():
                return True
            else:
                return None
            
        else:
            '''Moves for a point can be :
            - Random walk 
            - Random walk with the staying still possibility
            - No moves if no attributes are set
            '''
            if self.moving == "Random":
                new_position = self.random_move(impossible_positions)
                new_position = (new_position[0],new_position[1])
                self.set_position(new_position)
            elif self.moving == "Random still":
                new_position = self.random_move(impossible_positions,still = True)
                new_position = (new_position[0],new_position[1])
                self.set_position(new_position)
                
    def possible_position(self,position,impossible_positions):
        x,y = position
        if position in impossible_positions or x < 0 or y < 0 or x >= self.height or y >= self.width:
            return False
        else:
            return True
           
    '''-----------------------------------------------------------------------------------------'''
    '''PICKUP POINTS METHODS'''
    def random_move(self,impossible_positions = [],still = False):
        '''Random move method for a pickup point, returns no if no position is reachable
        if the parameter "still" is true, the point has the possibility of not moving and stay still in his choice
        '''
        possible_moves = self.possible_moves if still else [x for x in self.possible_moves if x != (0,0)]
        possible_positions = [tuple(sum(x) for x in zip(self.position,y)) for y in possible_moves]
        possible_positions = [x for x in possible_positions if self.possible_position(x,impossible_positions)]
        if len(possible_positions) > 0:
            return random.choice(possible_positions)
        else:
            return self.position
                
            
    '''-----------------------------------------------------------------------------------------'''
    '''AI AGENT METHODS'''
    def reach_goal(self):
        '''Returns True if the agent reaches his goal.
        This function has to be modified to implement multiple points'''
        if self.position in self.goal:
            return True
        else:
            return False
                
    def out_of_bounds(self):
        '''If the agent goes out of the environment returns False'''
        if self.x < 0 or self.y < 0 or self.x >= self.height or self.y >= self.width:
            return True
        else:
            return False
        
        











class Pickup_env():
    def __init__(self,size=(8,8),pickup_points = None,n_points = 1,time_step = 1,render = False,render_update = False,AI_position = None,alpha = 0.5,beta = 0.5,gamma = 5):
        '''INTIALIZATION
        An environment has : 
        - A size (height and width)
        - A running time step count
        - A running turn count, defining if it's the AI or the points turn to move on the grid (1 for the AI and -1 for the points)
        - Hyperparameters for rewards/punishments decay alpha and beta, and for max time limit gamma
        - pickup points that are either randomly set or predefined
        - An AI agent
        - A model which is a matrix representing the environment grid where 0 is nothing, 1 the AI and -1 a pickup point
        - render attribute works with the _print method which only prints if render is true
        - render_update will be useful to watch a full episode of training the agent by showing every steps. 
        '''
        
        '''RENDERING OPTIONS'''
        self.render = render
        self.render_update = render_update
        
        self._print('>> Environment initialization')
        self.size = size
        self.height,self.width = size
        self.time_step = time_step
        self.turn = 1
        
        '''HYPERPARAMETERS'''
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.time_limit = self.gamma*(self.height+self.width)
        
        '''PICKUP POINTS'''
        if pickup_points == None:
            self._print('.. No Pickup points, setting one randomly which does not move') #to change for multiple points
            self.pickup_points = [Point(size = self.size,moving = False) for x in range(0,n_points)]
        else:
            self.pickup_points = pickup_points
        '''ENVIRONMENT MODEL'''
        self.model = np.zeros(self.size)
        self.are_pickup_points_moving = False
        
        for point in self.pickup_points:
            if self.are_pickup_points_moving or point.moving != False:
                self.are_pickup_points_moving = True        
            self.model[point.position] = point.value
            
        pickup_positions =  self.find_pickup_points()
        
        '''AI AGENT'''
        self.AI = Point(self.size,moving = True,is_AI = True,no_positions = pickup_positions,goal = pickup_positions,position = AI_position)
        self.model[self.AI.position] = self.AI.value
        self.all_points = self.pickup_points+[self.AI]
        

        
    '''-----------------------------------------------------------------------------------------'''
    def __repr__(self):
        '''CONVENIENT REPRESENTATION'''
        return str(self.model).replace(' 0.','. ').replace('-1.','O ').replace(' 1.','X ')
    
    def __str__(self):
        '''CONVENIENT REPRESENTATION'''
        return str(self.model).replace(' 0.','. ').replace('-1.','O ').replace(' 1.','X ')
    
    def _print(self,string):
        if self.render:
            print(string)
            
            
    '''-----------------------------------------------------------------------------------------'''
    '''ENVIRONMENT METHODS'''
    def find_pickup_points(self):
        '''Returns a list of tuples of all the pickup points positions'''
        return [x for x in zip(*[list(x) for x in np.where(self.model == -1)])]
    
    def find_all_points(self):
        '''Returns a list of tuples of all the pickup points and AI positions
        Helps to define the positions in where a pickup point cannot go
        '''
        return [x for x in zip(*[list(x) for x in np.where(self.model != 0)])]
    

    
    def update(self,action = None):
        '''UPDATE FUNCTION
        If the points are moving, an update calls one turn : 
        - First the AI move according to the defined action
        - Then another update call is needed to move the points
        Thus 2 updates are needed to finish a "turn" where everyone has moved
        Otherwise, one update call is enough to run a turn. 
        '''
                                                 
                                                 
        '''AI TURN'''
        if self.turn == 1:
            for point in [self.AI]:
                self.model[point.position] = 0
                move = point.move(action = action)
                #reward function must change to valorize time step when largely negative and penalize when largely positive
                if move == False:
                    self._print('Out of bounds')
                    return self.time_step,float(-1)/self.time_step**self.beta 
                elif move == True:
                    self._print('Pickup point !')
                    self.model[point.position] = point.value
                    self.pickup_points = [x for x in self.pickup_points if x.position != point.position]
                    return self.time_step,float(1)/self.time_step**self.alpha
                else:
                    self.model[point.position] = point.value
                    if self.time_step > self.time_limit:
                        return self.time_step,float(-1)/self.time_step**self.beta
                self.time_step = self.time_step + 1
                
                if self.are_pickup_points_moving: self.turn = -1
        else:
            '''PICKUP POINTS TURN'''
            for point in self.pickup_points:
                self.model[point.position] = 0
                point.move(impossible_positions = self.find_all_points())
                self.model[point.position] = point.value
            
            self.AI.set_goal(self.find_pickup_points())
            self.turn = 1
            
        
def plot_running_reward(rewards,training_episodes):
    plt.figure(figsize = (15,5))
    plt.title('Reward after %s training episodes' % training_episodes)
    plt.plot(rewards)
    plt.plot([0 for x in rewards])
    plt.show()



def random_simulation(size = (8,8)):
    environment = Pickup_env(size = size)
    update = environment.update()
    while update == None:
        update = environment.update()
    return update

def random_simulations(size = (8,8),n_simulations = 100):
    steps_necessary = []
    rewards = []
    for _ in range(n_simulations):
        output = random_simulation(size = size)
        steps_necessary += [output[0]]
        rewards += [output[1]]
    return steps_necessary,rewards

def plot_simulations(size = (8,8),n_simulations = 100):
    print('%s simulations on a %s environment' % (n_simulations,size))
    X,Y = random_simulations(size,n_simulations)
    plt.figure(figsize=(15,5))
    plt.title('Number of time steps to reach the pickup goal')
    plt.hist(X,bins = 20)
    plt.show()
    plt.figure(figsize=(15,5))
    plt.title('Distribution of rewards')
    plt.hist(Y,bins = 20)
    plt.show()







try:
    import cPickle as pickle
except Exception as e:
    import pickle







class NeuralNetwork():
    def __init__(self,environment,H = 200,batch_size = 10,model = None):
        '''HYPERPARAMETERS'''
        self.H = H #number of hidden neurons
        self.batch_size = batch_size #every how many episodes to do a parameter update
        self.learning_rate = 1e-2
        self.gamma = 0.99 #discount factor for the reward
        self.decay_rate = 0.99 #decay factor for RMSprop leaky sum of grad^2
        #add resume and render ?
        
        self.environment = environment #loading the environment
        self.D = self.environment.width * self.environment.height
        
        '''CREATING A 2 LAYER NEURAL NETWORK'''
        if model != None:
            self.model = pickle.load(open(model+".p", 'rb'))
        else:
            self.model = {}
            self.model['W1'] = np.random.randn(self.H,self.D) / np.sqrt(self.D) # "Xavier" initialization
            self.model['W2'] = np.random.randn(5,self.H) / np.sqrt(self.H)
        

        # python2.7
        # self.grad_buffer = { k : np.zeros_like(v) for k,v in self.model.iteritems() } # update buffers that add up gradients over a batch
        # self.rmsprop_cache = { k : np.zeros_like(v) for k,v in self.model.iteritems() } # rmsprop memory
    

        # python 3.5
        self.grad_buffer = { k : np.zeros_like(v) for k,v in self.model.items() } # update buffers that add up gradients over a batch
        self.rmsprop_cache = { k : np.zeros_like(v) for k,v in self.model.items() } # rmsprop memory


    def load_model(self,model = "save"):
        self.model = pickle.load(open(model+".p", 'rb'))
    
    def set_learning_rate(self,learning_rate):
        self.learning_rate = learning_rate
        
    def sigmoid(self,x):
        return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]
    
    def softmax(self,x):
        x = x - np.max(x) #translating from the maximum to avoid overflow data and large number numeric approximation (more information in CS231n lecture 2)
        return np.exp(x)/np.sum(np.exp(x))
    
    def discount_rewards(self,r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def policy_forward(self,x):
        h = np.dot(self.model['W1'], x)
        h[h<0] = 0 # ReLU nonlinearity
        logp = np.dot(self.model['W2'], h)
        p = self.softmax(logp)
        return p, h

    def policy_backward(self,eph, epdlogp,epx):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(eph.T, epdlogp).T
        dh = np.dot(epdlogp, self.model['W2'])
        dh[eph <= 0] = 0 # backpro prelu
        dW1 = np.dot(dh.T, epx)
        return {'W1':dW1, 'W2':dW2}
    

class Game():
    def __init__(self,size = (8,8),pickup_points = None,AI_position = None,batch_size = 10,alpha = 0.5,beta = 0.5, gamma = 5):
        self.pickup_points = pickup_points
        self.AI_position = AI_position
        self.alpha,self.beta,self.gamma = alpha,beta,gamma
        self.parameters = {"alpha":self.alpha,"beta":self.beta,"gamma":self.gamma}
        self.env = Pickup_env(size = size,pickup_points = pickup_points,AI_position = AI_position,**self.parameters)
        self.NN = NeuralNetwork(self.env,batch_size = batch_size)
        self.prev_x = None
        self.xs,self.hs,self.dlogps,self.drs = [],[],[],[]
        self.running_reward = None
        self.reward_sum = 0
        self.episode_number = 0
        self.action = {(0,1):"RIGHT",(0,-1):"LEFT",(1,0):"DOWN",(-1,0):"UP",(0,0):"STILL"}
        
    def load_neural_network(self,NN):
        self.NN = NN
        
    def save_neural_network(self,model_name = "save"):
        pickle.dump(self.NN.model, open(model_name+".p", 'wb'))
        
    def preprocess(self):
        return self.env.model.astype(np.float).ravel()
    
    def sample_action(self,probabilities):
        possible_moves = self.env.AI.possible_moves
        choice = np.random.choice(5,p = np.array(probabilities))
        onehot_vector = np.zeros(5)
        onehot_vector[choice] = 1
        return possible_moves[choice],onehot_vector


    def animation(self,turn = "AI",time_step = 0.5):
        # sys.stdout.write('==> %s TURN\n'%turn + str(self.env))
        # sys.stdout.write('\r\r\r'+str(self.env))
        sys.stdout.writelines('\r'+str(np.random.randint(1,9))+'\n'+str(np.random.randint(1,9))+'\r')
        sys.stdout.flush()

        time.sleep(time_step)

    


    def run_step(self,show = False,record = True,render = False,animation = -1):
        if render and animation <= 0:
            print(self.env)
            print('==> AI TURN')
        elif render and animation > 0:
            self.animation("AI",animation)
            
        x = self.preprocess()
    
        # forward the policy network and sample an action from the returned probability
        probabilities, h = self.NN.policy_forward(x)
        action,onehot_vector = self.sample_action(probabilities)
        
        # record various intermediates (needed later for backprop)
        if record:
            self.xs.append(x) # observation
            self.hs.append(h) # hidden state
            self.dlogps.append(onehot_vector - probabilities) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
        
        # step the environment and get new measurements
        update = self.env.update(action)
        done = False if update == None else True
        reward = 0 if update == None else update[1]
        if record:
            self.reward_sum += reward
            self.drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
            
        if render and animation <= 0: 
            print("> Action sampled in %s : %s - %s" % ([round(x,4) for x in probabilities],action,self.action[action]))
            print("")
            
        if self.env.turn == -1:
            if render and animation <= 0:
                print(self.env)
                print('==> PICKUP POINTS TURN')
                print('')
            elif render and animation > 0:
                self.animation("PICKUP",animation)
            self.env.update()
            
            
        return done,reward,probabilities,action
    
    def run_episode(self,render = False,render_first = False,record = True,animation = -1):
        time_step = 1
        self.env = Pickup_env(size = self.env.size,pickup_points = self.pickup_points,AI_position = self.AI_position,**self.parameters)
        if not animation:
            if render_first and time_step == 1: print(self.env)
        done,reward,probabilities,action = self.run_step(record = record,render = render,animation = animation)
        first = [probabilities,action]
        
        while done == False:
            time_step += 1
            done,reward,probabilities,action = self.run_step(record = record,render = render,animation = animation)
            
        if render: print(">>> Episode finished in %s time steps - reward : %f " % (time_step,reward) + ('' if reward < 0 else ' !!!!!!!!'))
        return done,reward,probabilities,action,first
            
            
    def train(self,render_first = False,n_episodes = 100000,intermediary_steps = 1000,saving_model = False):
        i = 0
        rewards = []
        all_rewards = []
        episodes_trained = 0
        if type(intermediary_steps) == int:
            intermediary_steps_render = intermediary_steps
            intermediary_steps_record = int(intermediary_steps/10)
        elif type(intermediary_steps) == tuple:
            intermediary_steps_render = intermediary_steps[0]
            intermediary_steps_record = intermediary_steps[1]

        while episodes_trained < n_episodes:
            done,reward,probabilities,action,first = self.run_episode(render_first = (render_first and (self.episode_number) % intermediary_steps_render == 0))
            self.episode_number += 1
            episodes_trained += 1
            
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(self.xs)
            eph = np.vstack(self.hs)
            epdlogp = np.vstack(self.dlogps)
            epr = np.vstack(self.drs)
            self.xs,self.hs,self.dlogps,self.drs = [],[],[],[] # reset array memory
                    
            # compute the discounted reward backwards through time
            discounted_epr = self.NN.discount_rewards(epr)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            if sum(discounted_epr == 0) == len(discounted_epr):
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
    
            epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
            grad = self.NN.policy_backward(eph, epdlogp,epx)
            for k in self.NN.model: self.NN.grad_buffer[k] += grad[k] # accumulate grad over batch
            rewards.append(reward)
            if (self.episode_number) % intermediary_steps_render == 0:
                if render_first:
                    print("> First episode action sampled in %s : %s - %s" % ([round(x,4) for x in first[0]],first[1],self.action[first[1]]))

                if render_first: 
                    print('>>> ep %d: game finished, reward: %.4f' % (self.episode_number, reward) + ('' if reward == None or reward < 0 else ' !!!!!!!!'))
                    print('')
                else:
                    sys.stdout.write('\r'+'>>> ep %d: game finished, reward: %.4f' % (self.episode_number, reward) + ('         ' if reward == None or reward < 0 else ' !!!!!!!!'))
                    sys.stdout.flush()

            if (self.episode_number) % intermediary_steps_record == 0:
                all_rewards.append(sum(rewards)/intermediary_steps_record)
                rewards = []
                
            
            # perform rmsprop parameter update every batch_size episodes
            if self.episode_number % self.NN.batch_size == 0:
                for k,v in self.NN.model.items():
                    g = self.NN.grad_buffer[k] # gradient
                    self.NN.rmsprop_cache[k] = self.NN.decay_rate * self.NN.rmsprop_cache[k] + (1 - self.NN.decay_rate) * g**2
                    self.NN.model[k] += self.NN.learning_rate * g / (np.sqrt(self.NN.rmsprop_cache[k]) + 1e-5)
                    self.NN.grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

            # boring book-keeping
            self.running_reward = self.reward_sum if self.running_reward is None else self.running_reward * 0.99 + self.reward_sum * 0.01
            if (saving_model and self.episode_number % intermediary_steps == 0): self.save_neural_network()
            self.reward_sum = 0
            self.env = Pickup_env(size = self.env.size,pickup_points = self.pickup_points,AI_position = self.AI_position,**self.parameters)
            #self.prev_x = None
        
        return all_rewards
        

    
class Brain():
    def __init__(self):
        

