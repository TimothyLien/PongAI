import pygame
import random
import sys

from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import os


from stable_baselines3 import A2C

# creates the display for the screen
pygame.init()
screen = pygame.display.set_mode((500,500))

pygame.display.set_caption("Pong AI Game")


clock = pygame.time.Clock()


# Paddle class for the player
class Paddle:
    def __init__(self,pos):
        self.rect = pygame.Rect(pos[0],pos[1],5,100)
        self.vel = pygame.Vector2()
        self.vel.x,self.vel.y = 0,0
    def move(self,keys):
        if keys[pygame.K_w]:
            self.vel.y  = -5
        elif keys[pygame.K_s]:
            self.vel.y =  5
        else:
            self.vel.y  = 0 

        self.rect.y += self.vel.y 

        if self.rect.y < 0:
            self.rect.y = 0
        if self.rect.y + self.rect.height > 500:
            self.rect.y = 500-self.rect.height
    def coord_y(self):
        return self.rect.y
    def draw(self):
        pygame.draw.rect(screen,(255,255,255),self.rect)

# Paddle class for the AI
class PaddleAgent:
    def __init__(self,pos):
        self.rect = pygame.Rect(pos[0],pos[1],5,100)
        self.vel = pygame.Vector2()
        self.vel.x,self.vel.y = 0,0

    def coord_y(self):
        return self.rect.y
    def move(self,action):
        if action == 0:
            self.vel.y  = -5
        elif action == 1:
            self.vel.y =  5
        else:
            self.vel.y  = 0 
        if self.rect.y < 0:
            self.rect.y = 0
        if self.rect.y + self.rect.height > 500:
            self.rect.y = 500-self.rect.height
        self.rect.y += self.vel.y 
    def  draw(self):
        pygame.draw.rect(screen,(255,255,255),self.rect)

   
# Ball class
class Ball:
    def __init__(self,pos):
        self.col = (255,255,255)
        self.rect = pygame.Rect(pos[0],pos[1],20,20)
        self.vel = pygame.Vector2()
        self.vel.x, self.vel.y = 8, random.choice([-4,-3,-2,-1,0,1,2,3,4])
    def coord_x(self):
        return self.rect.x
    def coord_y(self):
        return self.rect.y
    def move(self, paddle, opt_paddle):
        c_p1, c_p2  = False,False
        if self.rect.y < 0:
            self.vel.y *= -1
        if self.rect.y + self.rect.height > 500:
            self.vel.y *= -1

        if self.rect.x < 0:
            self.vel.x *= -1
        if self.rect.x + self.rect.width > 500:
            self.vel.x *= -1
        self.rect.y += self.vel.y 
        self.rect.x += self.vel.x
    
        if self.rect.colliderect(paddle.rect):
            self.vel.x *= -1
            distance  = (40 + paddle.rect.y)  - self.rect.y
            if distance > 0: 
                self.vel.y = (distance/15+4)*-1
                            
            elif distance < 0:
                self.vel.y = (abs(distance)/15+4)
                
            else:
                self.vel.y = self.vel.y * -1
        if self.rect.colliderect(opt_paddle.rect):
            self.vel.x *= -1
            distance  = (40 + opt_paddle.coord_y())  - self.rect.y
            if distance > 0:   
                self.vel.y = (distance/15+4)*-1
                            
            elif distance < 0:
                self.vel.y = (abs(distance)/15+4)
                
            else:
                self.vel.y = self.vel.y * -1      
    def draw(self):
        pygame.draw.rect(screen,self.col,self.rect)


# This is the pong environment the agent learns in
class pong_env(Env):
    def __init__(self, agent):
        # create the game
        self.ball = Ball((250,250))
        self.agent = agent
        self.paddle = Paddle([35,200])
        self.state = [self.ball.coord_x(), self.ball.coord_y(), self.agent.coord_y()]
        self.playerScore = 0
        self.agentScore = 0
        self.font = pygame.font.SysFont("symbol",100)
        self.score_board = self.font.render(f"{self.playerScore} {self.agentScore}", True, (255,255,255))
        self.board = self.score_board.get_rect()
        self.board.center = (220, 100)
        # Actions, up, down, leave
        self.action_space = Discrete(3)
        # The x and y coordinate of the ball
        self.observation_space = Box(low = np.array([0,0,0]), high = np.array([500,500,500]))
        
    def step(self, action,keys):
        # Determines the movement of each element for each frame of the game
        self.paddle.move(keys) 
        self.agent.move(action)
        self.ball.move(self.agent,self.paddle)
        self.state = [self.ball.coord_x(), self.ball.coord_y(), self.agent.coord_y()]
        done = False
        
        # Calculates the reward
        reward = 0
        if (abs(self.ball.coord_y()-self.agent.coord_y()) > 10):
            reward = -5
        else:
            reward = 1
        if self.ball.rect.colliderect(self.agent.rect):
            reward += 5

        # Determines if game has ended 
        if self.ball.coord_x() > 460:
            self.playerScore += 1
            done = True

        if self.ball.coord_x() < 15:
            self.agentScore += 1
            done = True

        info = {}
        return np.array(self.state), reward, done, False,info
    def render(self):
        # Renders the screen for each frame
        screen.fill((0,0,0))
        self.score_board = self.font.render(f"{self.playerScore}   {self.agentScore}", True, (255,255,255))
        screen.blit(self.score_board, self.board)
        self.agent.draw()
        self.paddle.draw()
        self.ball.draw()
        pygame.display.update()
        clock.tick(60)


    def reset(self):
        # After the ball has touched on the walls, this function resets everthing to their original positions
        self.paddle.x, self.paddle.y = 35,200
        self.ball.rect.x, self.ball.rect.y = 250,250
        self.ball.vel.x, self.ball.vel.y =  8, random.choice([-4,-3,-2,-1,0,1,2,3,4])
        self.agent.rect.x, self.agent.rect.y = 460,200
        self.state = [self.ball.coord_x(), self.ball.coord_y(), self.agent.coord_y()]
        return np.array(self.state), {}
        

# This initializes the AI and the pong environmeent
agent = PaddleAgent([460,200])
env = pong_env(agent)

# This loads the AI that has already been trained
loaded_model = A2C.load("sb_models")


# Runs the game
run = True
state,_  = env.reset()
while run:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    keys = pygame.key.get_pressed()
    action,_  = loaded_model.predict(state)
    next_state,reward, done,_, info = env.step(action,keys)
    env.render()
    state = next_state
    if done:
        state,_ = env.reset()


