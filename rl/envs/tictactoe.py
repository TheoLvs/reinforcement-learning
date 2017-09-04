#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
REINFORCEMENT LEARNING
Tic Tac Toe

Started on the 06/06/2017


theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import sys
import random
import time

from IPython.display import clear_output
from tqdm import tqdm
from collections import Counter

# Deep Learning (Keras, Tensorflow)
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import SGD,RMSprop, Adam
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D,ZeroPadding2D,Conv2D
from keras.utils.np_utils import to_categorical






#===========================================================================================================
# CELL DEFINITION
#===========================================================================================================



class Cell(object):
    def __init__(self,value = 0):

        """
        A cell can either be empty, or be occupied by player 1 or 2
        - value = 0 for an empty cell
        - value = 1 for player 1
        - value = 2 for player 2
        """

        self.value = value



    def vectorize(self):
        """
        Returns: a one hot encoded vector (numpy array) of the cell status np.array([0,0,1]) for a value 2
        """
        return to_categorical(self.value,num_classes = 3)[0]



    def set_value(self,value):
        """
        Updates: the attribute value of the cell, will be call to change the status of a cell
        """
        self.value = value


    def get_value(self):
        """
        Returns: the attribute value of the cell
        """
        return self.value



    def __repr__(self):
        if self.value == 0:
            return "   "
        elif self.value == 1:
            return " x "
        else:
            return " o "




    def __str__(self):
        return self.__repr__()







#===========================================================================================================
# GRID DEFINITION
#===========================================================================================================



class Grid(object):
    def __init__(self):
        
        self.initialize_grid()





    def __repr__(self,probas = None):

        if probas is not None: 
            probas = probas.reshape((3,3))

        for i in range(3):
            print("|",end ="")
            for j in range(3):
                cell = self.grid[i,j]

                if cell.value == 0 and probas is not None:
                    proba_value = str(np.round(probas[i,j],2))
                    proba_value = "1.0" if proba_value.startswith("1") else proba_value[1:4]
                    proba_value = proba_value + "0" if len(proba_value) == 2 else proba_value
                    print(proba_value,end = "|")
                else:
                    print(self.grid[i,j],end = "|")
            print("")

        return "3x3 TicTacToe grid"




    def __str__(self):
        return self.__repr__()




    #--------------------------------------------------------------------------------------------------------
    # INITIALIZATION

    def initialize_grid(self):
        self.grid = np.full((3,3),Cell())
        for i in range(3):
            for j in range(3):
                self.grid[i,j] = Cell()


        self.vectorized_grid = self.vectorize(one_hot = False)
        


    #--------------------------------------------------------------------------------------------------------
    # SETTERS


    def set_value(self,position,value):
        self.grid[position].set_value(value)
        self.vectorized_grid = self.vectorize(one_hot = False)







    #--------------------------------------------------------------------------------------------------------
    # GETTERS

    def get_free_cells(self):
        return list(zip(*list(map(list,np.where(self.vectorized_grid == 0)))))


    def get_free_cells_in_rows(self):
        free_cells = self.get_free_cells()
        cells = {k:[x for x in free_cells if x[0] == k] for k in range(3)}
        return cells


    def get_free_cells_in_columns(self):
        free_cells = self.get_free_cells()
        cells = {k:[x for x in free_cells if x[1] == k] for k in range(3)}
        return cells


    def get_free_cells_in_diagonals(self):
        free_cells = self.get_free_cells()
        cells = {k:[x for x in free_cells if ((k == 0 and x[0] == x[1]) or (k == 1 and x[0] == 2 - x[1]))] for k in range(2)}
        return cells


    def get_free_cells_in_axis(self,axis):
        if axis == 0:
            return self.get_free_cells_in_columns()
        elif axis == 1:
            return self.get_free_cells_in_rows()
        else:
            return self.get_free_cells_in_diagonals()


    def get_occupied_cells(self):
        return list(zip(*list(map(list,np.where(self.vectorized_grid != 0)))))


    def get_number_occupied_cells(self):
        return (self.vectorized_grid != 0).sum()


    def get_sum_rows(self,player):
        return np.sum(self.vectorized_grid == player,axis = 1)

    def get_sum_columns(self,player):
        return np.sum(self.vectorized_grid == player,axis = 0)


    def get_sum_diagonals(self,player):
        return np.sum(np.vstack([np.diag(self.vectorized_grid == player),np.diag(np.fliplr(self.vectorized_grid == player))]),axis = 1)


    def get_sum_axis(self,player,axis):
        if axis == 0:
            return self.get_sum_columns(player = player)
        elif axis == 1:
            return self.get_sum_rows(player = player)
        else:
            return self.get_sum_diagonals(player = player)







    #--------------------------------------------------------------------------------------------------------
    # VECTORIZER



    def vectorize(self,one_hot = False):
        
        if one_hot:
            return to_categorical(self.vectorize(one_hot = False),num_classes = 3).reshape(3,3,3).reshape((1,27))
        else:
            vectorizer = np.vectorize(lambda x : x.get_value())
            return vectorizer(self.grid)









    #--------------------------------------------------------------------------------------------------------
    # EVENTS MANAGEMENT



    def is_grid_full(self):
        return (self.get_number_occupied_cells() == 9)



    def is_won(self):
        if 3 in np.concatenate((self.get_sum_rows(1),self.get_sum_columns(1),self.get_sum_diagonals(1))):
            return (2,-2)
        elif 3 in np.concatenate((self.get_sum_rows(2),self.get_sum_columns(2),self.get_sum_diagonals(2))):
            return (-2,2)
        else:
            return None


    def is_done(self):

        reward = self.is_won()

        if reward is not None:
            return reward
        else:
            if self.is_grid_full():
                return (-1,-1)
            else:
                return None












#===========================================================================================================
# AGENTS DEFINITION
#===========================================================================================================



#--------------------------------------------------------------------------------------------------------
# BASE AGENT CLASS

class Agent(object):
    def __init__(self):
        self.rewards = []
        self.episode_rewards = []


    def set_value(self,value):
        self.value = value


    def get_value(self):
        return self.value


    def get_other_value(self):
        return 1 if self.value == 2 else 2


    def add_reward(self,reward):
        self.episode_rewards.append(reward)



    def train(self):
        pass


    def record(self,*args,**kwargs):
        pass


    def record_episode(self):
        self.running_rewards.append(np.sum(self.episode_rewards))
        self.reset_episode()
        

    def reset_episode(self):
        self.episode_rewards = []






#--------------------------------------------------------------------------------------------------------
# HUMAN AGENT CLASS


class Human_Agent(Agent):
    def __init__(self):
        self.rewards = []
        self.episode_rewards = []
        self.running_rewards = []




    def predict(self,grid):
        move = input()
        move = tuple([int(x) for x in move.split(",")])

        if move in grid.get_occupied_cells():
            raise ValueError("Impossible move")

        return None,None,None,move









#--------------------------------------------------------------------------------------------------------
# PURE RANDOM AGENT CLASS


class AI_Random_Agent(Agent):
    def __init__(self):
        self.rewards = []
        self.episode_rewards = []
        self.running_rewards = []


    def predict(self,grid):
        move = random.choice(grid.get_free_cells())
        return None,None,None,move









#--------------------------------------------------------------------------------------------------------
# RULES BASE AGENT CLASS


class AI_Rules_Agent(Agent):
    def __init__(self):
        self.rewards = []
        self.episode_rewards = []
        self.running_rewards = []




    def find_move_on_value_by_risk(self,grid,value,risk):

        def find_move_on_value_by_risk_by_axis(grid,value,risk,axis):
            sum_axis_player = grid.get_sum_axis(player = value,axis = axis)
            if risk in sum_axis_player:
                plays = np.where(sum_axis_player == risk)[0]
                play = random.choice(plays)
                moves = grid.get_free_cells_in_axis(axis = axis)[play]

                if len(moves) > 0:
                    move = random.choice(moves)
                    return move
                else:
                    return None


        axis_choice = [0,1,2]
        random.shuffle(axis_choice)


        for axis in axis_choice:
            move = find_move_on_value_by_risk_by_axis(grid,value,risk,axis)
            if move is not None:
                return move

        return None





    def predict(self,grid):

        # At the first turn to play we chose either the corners or the center of the grid
        if grid.get_number_occupied_cells() == 0.0:
            move = random.choice([(0,0),(0,2),(2,0),(2,2),(1,1)])
            return None,None,None,move


        # ATTACK
        # If already two marks, make the third to win the game
        move = self.find_move_on_value_by_risk(grid,self.get_value(),risk = 2)
        if move is not None:
            return None,None,None,move


        # DEFENSE
        # If the opposite player has already two marks, place the third to block the winning coup
        move = self.find_move_on_value_by_risk(grid,self.get_other_value(),risk = 2)
        if move is not None:
            return None,None,None,move



        # ATTACK
        # If already one mark, make a second to improve your likelihood to win
        move = self.find_move_on_value_by_risk(grid,self.get_value(),risk = 1)
        if move is not None:
            return None,None,None,move




        # Random sampling
        move = random.choice(grid.get_free_cells())
        return None,None,None,move










#--------------------------------------------------------------------------------------------------------
# POLICY GRADIENTS REINFORCEMENT LEARNING AGENT CLASS


class AI_RL_Agent(Agent):
    def __init__(self,model = None,lr = 0.1,epsilon = 0.2,gamma = 0.5,H = 100):

        # PARAMETERS INITIALIZATION
        self.learning_rate = lr
        self.gamma = gamma
        self.epsilon = epsilon

        self.rewards = []
        self.episode_rewards = []
        self.running_rewards = []
        self.x = []
        self.actions = []
        self.probas = []

        # LOADING NEURAL NETWORK CORE MODEL
        if model is None:
            self.model = self.load_default_model(H = H)
        else:
            self.model = model





    def load_default_model(self,H = 100):


        # Define the model
        model = Sequential()

        # First hidden layer
        model.add(Dense(H, input_dim=27))
        model.add(Activation('relu'))

        # Second hidden layer
        model.add(Dense(H))
        model.add(Activation('relu'))

        # Final layer
        model.add(Dense(9))
        model.add(Activation('softmax'))
        
        # Define an optimizer
        sgd = SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

        # Compile the model
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        return model







    def sample_action(self,probas,epsilon = 0.2):
        probas = probas[0]
        if np.random.rand() < epsilon:
            probas[probas != 0.0] = 1.0/len(probas[probas != 0.0])
            choice = np.random.choice(range(len(probas)),p = probas)
        else:
            choice = np.random.choice(range(len(probas)),p = probas)
        return choice



    def action_to_position(self,action):
        return (int(action/3),int(action%3))

    def position_to_action(self,position):
        return position[0] * 3 + position[1]



    def predict(self,grid):
        
        # Get the input vector
        input_vector = grid.vectorize(one_hot = True)

        # Prediction
        probas = self.model.predict(input_vector)

        # Reset the probas for the already occupied cells
        for position in grid.get_occupied_cells():
            removed_action = self.position_to_action(position)
            probas[0][removed_action] = 0.0

        # Renormalize the probas
        probas = probas / np.sum(probas)

        # Sampling action with epsilon greedy approach
        action = self.sample_action(probas,epsilon = self.epsilon)

        # Convert to move
        move = self.action_to_position(action)

        # Store the last probabilities to display them on the grid
        self.displayed_probas = probas

        return input_vector,action,probas,move



    def record(self,x = None,action = None,proba = None,reward = None,override = False):
        if x is not None:
            self.x.append(x)

        if action is not None:
            self.actions.append(action)

        if proba is not None:
            self.probas.append(proba)

        if reward is not None:
            if override:
                self.episode_rewards[-1] = reward
            else:
                self.episode_rewards.append(reward)




    def discounting_rewards(self,r,normalization = True):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add

        if normalization:
            discounted_r = np.subtract(discounted_r,np.mean(discounted_r),casting = "unsafe")
            discounted_r = np.divide(discounted_r,np.std(discounted_r),casting = "unsafe")

        return discounted_r


    def discount_rewards(self,normalization = True):
        rewards = np.vstack(self.episode_rewards)
        return self.discounting_rewards(rewards,normalization)




    def record_episode(self):
        self.running_rewards.append(np.sum(self.episode_rewards))
        self.episode_rewards = self.discount_rewards(normalization = False)
        self.rewards.extend(self.episode_rewards)
        self.reset_episode()








    def train(self):
        self.actions = to_categorical(np.vstack(self.actions),num_classes = 9)
        self.probas = np.vstack(self.probas)
        self.rewards = np.vstack(self.rewards)
        self.x = np.vstack(self.x)


        self.targets = self.rewards * (self.actions - self.probas) + self.probas

        self.model.train_on_batch(self.x,self.targets)

        self.x,self.actions,self.probas,self.rewards = [],[],[],[]






        













#===========================================================================================================
# GAME DEFINITION
#===========================================================================================================




class Game(object):
    def __init__(self,agent1,agent2,verbose = 1):
        self.verbose = verbose
        self.agent1 = agent1
        self.agent2 = agent2

        self.agent1.set_value(1)
        self.agent2.set_value(2)


    def run_episode(self,verbose=None):

        if verbose is None: verbose = self.verbose

        # Define an empty grid
        grid = Grid()


        # Display the first step of the grid
        if verbose:
            print(grid)

        # Initialize a loop
        i = 0

        # Determine the first player
        agents = [self.agent1,self.agent2]
        first_player_draw = np.random.choice([1,2])
        second_player_draw = 2 if first_player_draw == 1 else 1
        first_player = agents[first_player_draw - 1]
        second_player = agents[second_player_draw - 1]


        # Loop
        while True:
            if verbose:
                time.sleep(0.4)

            if i % 2 == 0:
                player_turn,other_player_turn = first_player_draw,second_player_draw
                player,other_player = first_player,second_player
            else:
                player_turn,other_player_turn = second_player_draw,first_player_draw
                player,other_player = second_player,first_player

            if verbose: print(">> Player {}'s turn".format(player_turn))


            # Predict or play the correct move
            x,action,proba,move = player.predict(grid)

            # Display probabilities in the RL case
            if verbose and "displayed_probas" in dir(player):
                time.sleep(1)
                clear_output()
                print(grid.__repr__(probas = player.displayed_probas))
                print("")
                time.sleep(1)



            # Set your movement and place your tick
            grid.set_value(move,player.get_value())
            
            if verbose:
                if player.get_value() == 3:
                    input()
                clear_output()
                print(grid)


            reward = grid.is_done()
            
            if reward is not None:

                # Display the results
                if np.sum(reward) == 0.0:
                    winner = reward.index(2) + 1
                    if verbose: print(">>> Player {} has won !!".format(winner))
                else:
                    if verbose: print(">>> It's a draw !")

                # Store the final reward
                player.record(x = x,action = action,proba = proba,reward = reward[player_turn-1])
                other_player.record(reward = reward[other_player_turn-1],override = True)

                player.record_episode()
                other_player.record_episode()

                # Break the loop to end the episode
                break
            else:
                player.record(x = x,action = action,proba = proba,reward = 0)
                if verbose: print("")


            i += 1





    def run_n_episodes(self,n,batch_size = 16):

        for i in tqdm(range(n)):
            self.run_episode(verbose = 0)

            if i % batch_size == 0:
                self.agent1.train()
                self.agent2.train()



    def plot_running_rewards(self,player1 = False,player2 = False):

        if player1 or player2:
            plt.figure(figsize = (15,4))

            if player1:
                plt.plot(np.cumsum(self.agent1.running_rewards)/np.array(range(1,len(self.agent1.running_rewards)+1)),label = "Player 1")

            if player2:
                plt.plot(np.cumsum(self.agent2.running_rewards)/np.array(range(1,len(self.agent2.running_rewards)+1)),label = "Player 2")

            plt.legend()
            plt.show()



    def analyze_rewards_set(self,x):
        length = len(x)
        count = Counter(x)
        games_won = count[2]/float(length)
        games_draw = count[-1]/float(length)
        games_lost = count[-2]/float(length)
        return games_won,games_draw,games_lost



    def analyze_rolling_rewards(self,rewards,rollback = 10):
        output = {"won":[],"draw":[],"won_draw":[],"lost":[]}
        for i in range(len(rewards)-rollback+1):
            x = rewards[i:i+rollback]
            won,draw,lost = self.analyze_rewards_set(x)
            output["won"].append(won)
            output["draw"].append(draw)
            output["lost"].append(lost)
            output["won_draw"].append(won + draw)
            
        return output



    def plot_results_during_training(self,rewards,rollback = 10,fields = ["won","draw","lost"]):

        plt.figure(figsize = (15,4))
        plt.title("Results in % for {} consecutive games".format(rollback))

        output = self.analyze_rolling_rewards(rewards,rollback)
        output = {k:v for k,v in output.items() if k in fields}

        possible_fields = ["won","draw","won_draw","lost"]
        fields = [k for k in possible_fields if k in fields]
        for k in fields:
            plt.plot(output[k],label = "% {}".format(k))

        plt.legend()
        plt.ylim([0,1])
        plt.show()
