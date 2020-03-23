"""Pygame test for multi agent modeling

Tutorials
https://zestedesavoir.com/tutoriels/846/pygame-pour-les-zesteurs/1381_a-la-decouverte-de-pygame/creer-une-simple-fenetre-personnalisable/#1-15425_creons-une-fenetre-basique
https://www.pygame.org/docs/ref/rect.html#pygame.Rect.move_ip
https://stackoverflow.com/questions/32061507/moving-a-rectangle-in-pygame 


Ideas:
- Add circles
- Pathfinding algorithm
- Obstacles
- Colliders
- Clicking to add agent or wall
- Grid
- AutoMaze
- Raytracing
- Change Icon
- Heatmaps of where agents were located (for retail purposes)

Projects:
- Epidemiology
- See MESA or NetLogo examples
- Bunny & Rabbits
"""

import numpy as np
import pygame
import time
import uuid

# import os
# os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (320,240)

pygame.init()
pygame.display.set_caption("Multi Agent Modeling Environment")
# ecran = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

screen = pygame.display.set_mode((1000, 600))

simulation_on = True
# time.sleep(5)

background_colour = (0, 0, 0)






class RectangleAgent:

    def __init__(self,width,height,x,y,screen = None):
        # Rect left top width height

        self.screen = screen
        self.fig = pygame.rect.Rect((x,y,width,height))
        # print(f"Initialized rect at {self.pos}")

        self.change_direction()

        self.agent_id = str(uuid.uuid1())


    @property
    def pos(self):
        return self.fig.x,self.fig.y,self.fig.width,self.fig.height

    def move_at(self,x,y):
        self.x = x
        self.y = y

    
    def wander(self,dl):

        self.move(angle = self.direction_angle,dl = dl)


    def change_direction(self):
        self.direction_angle = np.random.uniform(0,2*np.pi)


    def move_towards(self):
        pass


    def collides(self,agents):

        if len(agents) == 0:
            collisions = []
        else:
            other_agents = [agent.fig for agent in agents if agent.agent_id != self.agent_id]
            collisions = self.fig.collidelistall(other_agents)

        if len(collisions) > 0:
            return True,collisions
        else:
            return False,collisions


    def if_collides(self,agents):

        is_collision,collisions = self.collides(agents)

        if is_collision:
            self.direction_angle += np.pi

    

    def move(self,dx = 0,dy = 0,angle = None,dl = None,colliders = None):

        if angle is not None:
            assert dl is not None

            # Compute delta directions with basic trigonometry
            dx = dl * np.cos(angle)
            dy = dl * np.sin(angle)
            self.move(dx = dx,dy = dy)

        else:
    
            screen_width = self.screen.get_width()
            screen_height = self.screen.get_height()

            old_x = self.fig.x
            old_y = self.fig.y

            if self.fig.x + dx > screen_width:
                self.fig.x = 0
            elif self.fig.x + dx < 0:
                self.fig.x = screen_width
            else:
                self.fig.x = self.fig.x + dx

            if self.fig.y + dy > screen_height:
                self.fig.y = 0
            elif self.fig.y + dy < 0:
                self.fig.y = screen_height
            else:
                self.fig.y = self.fig.y + dy

        if colliders is not None:
            if self.collides(colliders):
                self.fig.x = old_x
                self.fig.y = old_y


        # print(f"Position at {self.fig.x},{self.fig.y}")


    def render(self,color = (180,20,150)):
        pygame.draw.rect(self.screen,color,self.pos)
        # pygame.draw.circle(self.screen,color,(self.fig.x,self.fig.y),10)
        # pass




class Obstacle:
    def __init__(self,width,height,x,y,screen = None):
        # Rect left top width height

        self.screen = screen
        self.fig = pygame.rect.Rect((x,y,width,height))
        # print(f"Initialized rect at {self.pos}")
        self.agent_id = str(uuid.uuid1())


    def render(self,color = (10,150,10)):
        pygame.draw.rect(self.screen,color,self.pos)


    @property
    def pos(self):
        return self.fig.x,self.fig.y,self.fig.width,self.fig.height



size = 10
n_rects = 500

rects = []

for i in range(n_rects):
    new_rect = RectangleAgent(
        size,size,
        np.random.uniform(0,screen.get_width()),
        np.random.uniform(0,screen.get_height()),
        screen,
    )

    rects.append(new_rect)


    

i = 0
stop = 1000

obstacles = [
    Obstacle(200,200,300,300,screen)
]


while simulation_on:
    screen.fill(background_colour)

    for rect in rects:
        rect.wander(size)
        rect.if_collides(rects + obstacles)

    for rect in rects + obstacles:
        rect.render()

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            simulation_on = False

        elif event.type == pygame.MOUSEBUTTONUP:
            new_x,new_y = pygame.mouse.get_pos()
            # new_rect = RectangleAgent(size,size,new_x,new_y,screen)
            # rects.append(new_rect)

            new_obs = Obstacle(20,20,new_x,new_y,screen)
            obstacles.append(new_obs)



    pygame.display.update()
    # pygame.display.flip()

    time.sleep(0.05)


    if i == stop:
        simulation_on = False
    else:
        i+=1


pygame.quit()