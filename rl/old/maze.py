#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
REINFORCEMENT LEARNING
Maze environment and development


Started on the 04/01/2017



theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


import numpy as np
# import gym
import os
import time

from keras.models import load_model, Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, RMSprop


from ai.rl.algorithms import Brain



class Cell():
    def __init__(self,type = "empty",purpose = "",occupied = False):

        if type not in ["empty","wall"]:
            raise ValueError("Unknown ype for the cell")
        if purpose not in ["","start","end"]:
            raise ValueError("Unknown purpose of the cell")

        self.type = type

        if type == "empty":
            self.occupied = occupied
            self.value = -1 if not self.occupied else 0
            self.value = 2 if purpose == "end" else -1

        elif type == "wall":
            self.occupied = False
            self.value = 1


        self.way = 0


        self.purpose = purpose

        self.vectorize()


    def vectorize(self):
        self.vector = np.zeros(3,dtype = float)
        if self.value >= 0:
            self.vector[self.value] = 1.0


    def switch_status(self):
        if self.type == "empty":
            if self.occupied:
                self.occupied = False
                self.value = -1
                self.vectorize()
            else:
                self.occupied = True
                self.value = 0
                self.way += 1
                self.vectorize()

    def set_purpose(self,purpose):
        self.__init__(type = self.type,purpose = purpose)


    def __repr__(self):
        if self.value == -1:
            return "empty"
        elif self.value == 0:
            return "agent"
        elif self.value == 1:
            return "wall"
        elif self.value == 2:
            return "end"

    def __str__(self):
        return self.__repr__()














class Maze():
    def __init__(self,grid = [],starts = [],ends = [],deadline = 0,tolerance = 0.8):
        if len(grid) == 0:
            self.grid = self.create_random_grid(size)
        else:
            self.grid = deepcopy(grid)
            # if len(self.grid.shape) == 1:
            #     self.grid = np.reshape(self.grid,size)


            for line in self.grid:
                for cell in line:
                    cell.occupied = False
                    cell.way = 0


        self.shape = self.grid.shape
        self.original_starts = starts
        self.original_ends = ends

        self.reset_grid()

        self.starts = starts if len(starts) > 0 else self.find_empty_cells()
        self.ends = ends if len(ends) > 0 else self.find_empty_cells()

        self.start = self.starts[np.random.choice(len(self.starts))]
        ends = [x for x in self.ends if x != self.start]
        self.end = ends[np.random.choice(len(ends))]

        self.grid[self.start].set_purpose("start")
        self.grid[self.end].set_purpose("end")

        self.moment = 0

        self.tolerance = tolerance


        self.agent = Agent(start_position = self.start)
        self.switch_cell_status(self.start)

        self.actions_moves = {0:"UP",1:"RIGHT",2:"DOWN",3:"LEFT"}
        self.moves_actions = {v:k for k,v in self.actions_moves.items()}

        if deadline == 0:
            self.deadline = int((self.count_empty_cells() - 1)*self.tolerance)

        else:
            self.deadline = deadline

        self.positive_reward = (self.count_empty_cells() - 1)

    def reset_grid(self):
        for line in self.grid:
            for cell in line:
                cell.set_purpose("")


    def vectorize(self):
        return np.array(list(map(lambda x: x.vector,self.grid.reshape(1,self.shape[0]*self.shape[1])[0]))).reshape(1,self.shape[0]*self.shape[1]*3)[0]


    def action_to_vector(self,action):
        if type(action) == str:
            action = self.moves_actions[action]
        vector = np.zeros(4)
        vector[action] = 1.0
        return vector

    def vector_to_action(self,vector):
        return int(np.where(vector == 1)[0][0])




    def create_random_grid(self,size):
        grid = []
        return grid


    def show(self):
        r = ""
        r += "_"*(self.shape[1]) + "\n"
        for i,line in enumerate(self.grid):
            for cell in line:
                if cell.occupied:
                    r += "X"
                elif cell.purpose == "end":
                    r += "O"
                elif cell.type == "empty":
                    r += " "
                elif cell.type == "wall":
                    r += "#"
            r += "\n"
        r += "¯"*(self.shape[1])
        return r

    def __repr__(self):
        return self.show()

    def __str__(self):
        return self.show()

    def render(self):
        print(self.show())



    def find_in_grid(self,func):
        func = np.vectorize(func,otypes = [bool])
        return tuple(np.array(np.where(func(self.grid) == True)).reshape(1,2)[0])

    def find_purpose(self,purpose = "start"):
        func = lambda x : x.purpose == purpose
        return self.find_in_grid(func)

    def step(self,action):
        self.moment += 1
        reward = self.move_agent(action)
        done = reward == self.positive_reward
        observation = self.vectorize()
        info = {}

        return observation,reward,done,info


    def reset(self):
        self.__init__(self.grid,starts = self.original_starts,ends = self.original_ends,deadline = self.deadline, tolerance = self.tolerance)
        return self.vectorize()




    def possible_moves(self,position = []):
        if len(position) == 0:
            position = self.agent.position

        x,y = position
        moves = []

        # UP MOVE
        if x > 0 and self.grid[x-1,y].type == "empty":
            moves.append(1.0)
        else:
            moves.append(0.0)

        # RIGHT MOVE
        if y < self.shape[1] - 1 and self.grid[x,y+1].type == "empty":
            moves.append(1.0)
        else:
            moves.append(0.0)

        # DOWN MOVE
        if x < self.shape[0] - 1 and self.grid[x+1,y].type == "empty":
            moves.append(1.0)
        else:
            moves.append(0.0)

        if y > 0 and self.grid[x,y-1].type == "empty":
            moves.append(1.0)
        else:
            moves.append(0.0)

        return moves






    def switch_cell_status(self,position):
        self.grid[position].switch_status()


    def move_agent(self,action):
        """action can either be a vectorized move or the name"""
        if type(action) == str:
            action = self.moves_actions[action]


        vector = self.action_to_vector(action)
        possible_moves = self.possible_moves(self.agent.position)

        move = np.dot(vector,possible_moves)

        if move != 0.0:
            self.switch_cell_status(self.agent.position)
            self.agent.move(action)
            self.switch_cell_status(self.agent.position)
            new_possible_moves = self.possible_moves(self.agent.position)
            deadend = len(np.where(np.array(new_possible_moves) == 1)[0]) == 1

        else:
            raise ValueError("Impossible move")



        if self.agent.position == self.end:
            return self.positive_reward

        elif self.moment == self.deadline:
            return -self.positive_reward
        elif deadend:
            return -1
        else:
            return 0



    def random_action(self):
        possible_moves = self.possible_moves(self.agent.position)
        possible_moves /= np.sum(possible_moves)
        action = np.random.choice(len(possible_moves),p = np.array(possible_moves))

        return action


    def count_empty_cells(self):
        return len(self.find_empty_cells())

    def find_empty_cells(self):
        empty_cell = np.vectorize(lambda x : x.type == "empty" and x.purpose == "")
        cells = empty_cell(self.grid)     
        return list(map(tuple,list(np.array(np.where(empty_cell(self.grid) == True)).T)))

    def select_two_random_cells(self):
        cells = self.find_empty_cells()
        start,end = np.random.choice(len(cells),2,replace = False)
        return cells[start],cells[end]








class Agent():
    def __init__(self,start_position):
        self.position = start_position

    def move(self,action):
        x,y = self.position

        if action == 0:
            self.position = (x-1,y)
        elif action == 1:
            self.position = (x,y+1)
        elif action == 2:
            self.position = (x+1,y)
        elif action == 3:
            self.position = (x,y-1)




import numpy as np
import pygame
from pygame import locals as pygame_const
import time

from copy import deepcopy



class Game():
    def __init__(self,grid,starts = [],ends = [],reload = True,deadline = 0,tolerance = 0.8):

        self.maze = Maze(grid,starts,ends,deadline,tolerance)
        size = self.maze.shape
        self.brain = Brain(self.maze,"Maze",reload = reload,input_dim = size[0]*size[1]*3,output_dim = 4)

        self.deadline = self.maze.deadline



        height = 640
        width = 900

        cell_size = int(height/self.maze.shape[0])


        self.pygame = {
            "cell_size":cell_size,
            "height":height,
            "width":width,
            "grid_width":height
        }





    def run_episode(self,n_steps = 0,move = "RL",render = False,record = False,pygame = True,intermediary_print = 0):
        if intermediary_print == 0:
            intermediary_print = self.brain.batch_size

        if n_steps == 0:
            # n_steps = self.maze.shape[0]*self.maze.shape[1]*self.maze.zeta
            n_steps = self.deadline

        observation = self.maze.reset()

        if render and pygame:
            self.pygame_init()

        on = True
        simulation = True

        while simulation:

            self.reward_sum = 0
            step = 0
            while on:
                # print(self.maze.vectorize().reshape(self.maze.shape[0],self.maze.shape[1],4))
                step += 1


                if move == "random":

                    if render:
                        if pygame: 
                            self.pygame_update()
                        else:
                            self.maze.render()
                    action = self.maze.random_action()
                    observation,reward,done,info = self.maze.step(action)
                    self.reward_sum += reward

                    if done or step == n_steps:
                        victory = "!!!!!!" if done else "       "
                        print('Episode finished in %s timesteps, reward : %s %s'%(i,running_reward,victory))
                        on = False
                        break

                else:
                    # print("step {} {} - {}".format(step,self.brain.running_reward,self.brain.reward_sum))

                    x,action,proba = self.brain.predict(observation,self.maze.possible_moves())

                    if not record:
                        # print(proba)
                        pass

                    if render:
                        if pygame: 
                            self.pygame_update(proba)
                        else:
                            self.maze.render()

                    observation,reward,done,info = self.maze.step(action)
                    action = self.brain.vectorize_action(action)

                    self.reward_sum += reward
                    if record:
                        self.brain.record(input = x,action = action,proba = proba,reward = reward)
                    if done or step == n_steps:
                        if record:
                            self.brain.record_episode()
                            if done:
                                victory = "!!!!!!"
                                self.wins += 1
                            else:
                                victory = "      "
                            # print("\rEpisode {}  {}/{}: total reward was {:0.03f} and running mean {:0.03f} {}".format(self.brain.episode_number, self.wins,self.brain.batch_size, self.brain.reward_sum, self.brain.running_reward,victory),end = "")
                            # print("step {} {} - {}".format(step,self.brain.running_reward,self.brain.reward_sum))
                            running_reward = self.brain.running_reward
                            if self.brain.episode_number % intermediary_print == 0:
                                print("Episode {}  {}/{}: total reward was {:0.03f} and running mean {:0.03f} {}".format(self.brain.episode_number, self.wins,self.brain.batch_size, self.brain.reward_sum, self.brain.running_reward,victory)) #,end = "  ")
                            
                            if self.brain.episode_number % self.brain.batch_size == 0:
                                self.brain.update_on_batch(show = False)
                                self.wins = 0

                            if self.brain.episode_number % 100 == 0:
                                self.brain.save_model()

                            self.brain.reset_episode()
                        else:
                            victory = "!!!!!!" if done else "      "
                            print("\rEpisode {} : total reward was {:0.03f} {}".format(self.brain.episode_number, reward, victory),end = "")

                        if render and pygame:
                            time.sleep(0.5)
                            self.pygame_update()
                            on,simulation = self.pygame_wait_until_something()
                        else:
                            on,simulation = False,False

                        break


                    if render and pygame:
                        on = self.pygame_events()
                        time.sleep(0.5)


        if render and pygame:
            self.pygame_exit()
        else:
            return running_reward







    def train(self,n_episodes,move="RL",intermediary_print = 0):

        rewards = []
        self.wins = 0
        for i in range(n_episodes):
            rewards.append(self.run_episode(move = move,render = False,record = True,intermediary_print = intermediary_print))


        return rewards


    def position(self,i,j,rectangle = True):
        if rectangle:
            return j*self.pygame['cell_size'],i*self.pygame['cell_size']
        else:
            pos = j*self.pygame['cell_size'] + int(self.pygame['cell_size']/2),i*self.pygame['cell_size'] + int(self.pygame['cell_size']/2)
            return pos


    def pygame_init(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.pygame['width'],self.pygame['height']))
        self.screen.fill((255,255,255))
        pygame.draw.rect(self.screen,(0,73,98),(self.pygame['grid_width'],0,self.pygame['width']-self.pygame['grid_width'],self.pygame['height']))



        pygame.draw.rect(self.screen,(255,255,255),(740,200,60,60))
        pygame.draw.rect(self.screen,(255,255,255),(740,320,60,60))
        pygame.draw.rect(self.screen,(255,255,255),(680,260,60,60))
        pygame.draw.rect(self.screen,(255,255,255),(800,260,60,60))

        # initialize font; must be called after 'pygame.init()' to avoid 'Font not Initialized' error
        myfont = pygame.font.SysFont("Arial", 40)

        label = myfont.render("Maze", 1, (255,255,255))
        self.screen.blit(label, (730, 20))

        reward_title = pygame.font.SysFont("Arial", 30).render("Reward", 1, (255,255,255))
        self.screen.blit(reward_title, (725,450))

        reward = pygame.font.SysFont("Arial", 30).render(" 0", 1, (255,255,255))
        self.screen.blit(reward, (755,500))





        pygame.display.set_caption("MAZE")

    def pygame_exit(self):
        pygame.display.quit()
        pygame.quit()



    def pygame_update(self,p = []):
        pygame.display.flip()
        # pygame.time.Clock().tick(30)

        for i,line in enumerate(self.maze.grid):
            for j,cell in enumerate(line):
                if cell.type == "wall":
                    pygame.draw.rect(self.screen,(0,0,0),(*self.position(i,j),self.pygame['cell_size'],self.pygame['cell_size']))
                elif cell.occupied:
                    pygame.draw.rect(self.screen,(int(np.max([255-50*cell.way,20])),int(np.max([255-50*cell.way,20])),255),(*self.position(i,j),self.pygame['cell_size'],self.pygame['cell_size']))
                    pygame.draw.circle(self.screen,(0,73,98),self.position(i,j,False),int(self.pygame['cell_size']/3),0)
                elif cell.purpose == "end":
                    pygame.draw.rect(self.screen,(0,123,164),(*self.position(i,j),self.pygame['cell_size'],self.pygame['cell_size']))
                elif cell.type == "empty" :
                    pygame.draw.rect(self.screen,(int(np.max([255-50*cell.way,20])),int(np.max([255-50*cell.way,20])),255),(*self.position(i,j),self.pygame['cell_size'],self.pygame['cell_size']))

        def convert_p(x):
            p = str(round(x,3))
            if len(p) == 3:
                p += "00"
            return p

        def convert_reward(x):
            if x >= 0:
                x = str(x)
                x = " "*(2-len(x))+x
            return str(x)

        if len(p) > 0:
            colors = list(map(lambda x: int(255*(1-x)),p))
            p = list(map(lambda x: convert_p(x),p))


            myfont = pygame.font.SysFont("Arial", 25)



            # UP
            pygame.draw.rect(self.screen,(255,colors[0],colors[0]),(740,200,60,60))
            label = myfont.render(p[0], 1, (0,0,0))
            self.screen.blit(label, (745, 215))

            # RIGHT
            pygame.draw.rect(self.screen,(255,colors[1],colors[1]),(800,260,60,60))
            label = myfont.render(p[1], 1, (0,0,0))
            self.screen.blit(label, (805, 275))

            # DOWN
            pygame.draw.rect(self.screen,(255,colors[2],colors[2]),(740,320,60,60))
            label = myfont.render(p[2], 1, (0,0,0))
            self.screen.blit(label, (745, 335))

            # LEFT
            pygame.draw.rect(self.screen,(255,colors[3],colors[3]),(680,260,60,60))
            label = myfont.render(p[3], 1, (0,0,0))
            self.screen.blit(label, (685, 275))


        pygame.draw.rect(self.screen,(0,73,98),(750,490,60,100))
        reward = pygame.font.SysFont("Arial", 30).render(convert_reward(self.reward_sum), 1, (255,255,255))
        self.screen.blit(reward, (755,500))

        pygame.display.update()


    def pygame_events(self):
        on = True
        for event in pygame.event.get():
            if event.type == pygame_const.KEYDOWN:
                if event.key == pygame_const.K_ESCAPE:
                    on = False
                if event.key == pygame_const.K_BACKSPACE:
                    observation = self.maze.reset()
                    # time.sleep(0.5)
                if event.key == pygame_const.K_SPACE:
                    self.pygame_wait_until_resume()

        return on 

    def pygame_wait_until_resume(self):
        wait = True
        while wait:
            for event in pygame.event.get():
                if event.type == pygame_const.KEYDOWN:
                    if event.key == pygame_const.K_SPACE:
                        wait = False
                        break

    def pygame_wait_until_something(self):
        on = True
        simulation = False
        wait = True
        while wait:
            for event in pygame.event.get():
                if event.type == pygame_const.KEYDOWN:

                    if event.key == pygame_const.K_ESCAPE:
                        wait = False
                        on = False
                    if event.key == pygame_const.K_BACKSPACE:
                        observation = self.maze.reset()
                        # time.sleep(0.5)
                        wait = False
                        simulation = True

        return on,simulation









"""--------------------------------------------------------------------
GRIDS
   --------------------------------------------------------------------
"""




toy_grid = np.array([
        Cell(purpose="end"),Cell("wall"),Cell(),Cell(),Cell("wall"),Cell(),Cell(),Cell(),
        Cell(),Cell("wall"),Cell(),Cell("wall"),Cell("wall"),Cell("wall"),Cell(),Cell("wall"),
        Cell(),Cell("wall"),Cell(),Cell(),Cell("wall"),Cell(),Cell(),Cell(),
        Cell(),Cell(),Cell("wall"),Cell(),Cell(),Cell(),Cell("wall"),Cell(),
        Cell("wall"),Cell(),Cell(),Cell(),Cell("wall"),Cell("wall"),Cell(),Cell(),
        Cell(),Cell("wall"),Cell("wall"),Cell(),Cell(),Cell(),Cell(),Cell("wall"),
        Cell(),Cell(),Cell(),Cell("wall"),Cell(),Cell("wall"),Cell(),Cell("wall"),
        Cell(),Cell("wall"),Cell(),Cell(),Cell(),Cell("wall"),Cell(),Cell(purpose = "start"),

    ]).reshape(8,8)



simple_grid = np.array([
        Cell(purpose="end"),Cell(),Cell(),
        Cell(),Cell("wall"),Cell(),
        Cell(),Cell(),Cell(purpose = "start"),
    ]).reshape(3,3)


simple_grid2 = np.array([
        Cell(purpose="end"),Cell(),Cell(),Cell(),
        Cell(),Cell("wall"),Cell(),Cell(),
        Cell(),Cell(),Cell(),Cell(),
        Cell(),Cell(),Cell(),Cell(purpose = "start"),
    ]).reshape(4,4)


simple_grid3 = np.array([
        Cell(purpose="end"),Cell(),Cell(),Cell(),Cell(),Cell(),
        Cell(),Cell("wall"),Cell(),Cell(),Cell(),Cell(),
        Cell(),Cell(),Cell(),Cell(),Cell(),Cell(),
        Cell(),Cell(),Cell(),Cell("wall"),Cell(),Cell(),
        Cell(),Cell(),Cell(),Cell(),Cell(),Cell(),
        Cell(),Cell(),Cell(),Cell(),Cell(),Cell(purpose = "start"),
    ]).reshape(6,6)


corridor = np.array([
        Cell("wall"),Cell("wall"),Cell("wall"),Cell(),Cell("wall"),
        Cell(purpose="end"),Cell(),Cell(),Cell(purpose = "start"),Cell(),
        Cell("wall"),Cell("wall"),Cell("wall"),Cell(),Cell("wall"),
        Cell("wall"),Cell("wall"),Cell("wall"),Cell("wall"),Cell("wall"),
        Cell("wall"),Cell("wall"),Cell("wall"),Cell("wall"),Cell("wall"),
        
    ]).reshape(5,5)








class Generator():
    def __init__(self,name = "empty",size = 4,force = True):

        self.size = size
        self.len = size**2


        #--------------------------------------------------------------------------------------

        if name == "empty":
            self.grid = np.array([Cell() for x in range(self.len)]).reshape(self.size,self.size)

            if force:
                self.starts = [(0,0)]
                self.ends = [(self.size-1,self.size-1)]


        elif name == "cross":
            if self.size %2 != 1: #size has to be an odd number
                raise ValueError("Size has to be an odd number")
            else:
                self.grid = np.array([Cell("wall") for x in range(self.len)]).reshape(self.size,self.size)
                middle = int(self.size/2)
                for cell in self.grid[:,middle]:
                    cell.__init__("empty")
                for cell in self.grid[middle,:]:
                    cell.__init__("empty")

            if force:
                self.starts = [(middle,middle)]
                self.ends = [(0,middle),(self.size-1,middle),(middle,0),(middle,self.size-1)]




        # start = np.random.choice(self.starts)
        # end =np.random.choice(self.ends)

        # self.grid[start].set_purpose("start")
        # self.grid[end].set_purpose("end")









