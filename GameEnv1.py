"""
Name: Priyanshu Ranka (NUID: 002305396)
Project: DDQN for Car Racing
Course: CS 5180 - Reinforcement Learning
Professor: Prof. Robert Platt
Semester: Spring 2025
Description: This file contains the implementation of the Car Racing environment for square track using Pygame.
"""

# Importing Libraries
import pygame
import math
from WallsSquare import Wall
from WallsSquare import getWalls
from GoalsSquare import Goal
from GoalsSquare import getGoals

# Constants for the game environment
GOALREWARD = 1
LIFE_REWARD = 0
PENALTY = -1

# Distance function - Square root of the sum of squares of the differences in x and y coordinates
def distance(pt1, pt2):
    return(((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2)**0.5)

# Rotate a point around an origin by a given angle
def rotate(origin,point,angle):
    qx = origin.x + math.cos(angle) * (point.x - origin.x) - math.sin(angle) * (point.y - origin.y)
    qy = origin.y + math.sin(angle) * (point.x - origin.x) + math.cos(angle) * (point.y - origin.y)
    q = myPoint(qx, qy)
    return q

# Rotate a rectangle defined by four points around its center by a given angle
def rotateRect(pt1, pt2, pt3, pt4, angle):

    pt_center = myPoint((pt1.x + pt3.x)/2, (pt1.y + pt3.y)/2)
    pt1 = rotate(pt_center,pt1,angle)
    pt2 = rotate(pt_center,pt2,angle)
    pt3 = rotate(pt_center,pt3,angle)
    pt4 = rotate(pt_center,pt4,angle)

    return pt1, pt2, pt3, pt4

# The myPoint class represents a point in 2D space with x and y coordinates.
class myPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# The myLine class represents a line segment defined by two points.  
class myLine:
    def __init__(self, pt1, pt2):
        self.pt1 = myPoint(pt1.x, pt1.y)
        self.pt2 = myPoint(pt2.x, pt2.y)

# The Ray class represents a ray originating from a point at a specific angle, with a method to cast the ray and check for intersections with walls.
class Ray:
    def __init__(self,x,y,angle):
        self.x = x
        self.y = y
        self.angle = angle

    # Cast method to check for intersections with a wall
    def cast(self, wall):
        x1 = wall.x1 
        y1 = wall.y1
        x2 = wall.x2
        y2 = wall.y2
        vec = rotate(myPoint(0,0), myPoint(0,-1000), self.angle)
        
        x3 = self.x
        y3 = self.y
        x4 = self.x + vec.x
        y4 = self.y + vec.y
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            
        if(den != 0):
            
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

            if t > 0 and t < 1 and u < 1 and u > 0:
                pt = myPoint(math.floor(x1 + t * (x2 - x1)), math.floor(y1 + t * (y2 - y1)))
                return(pt)

# The Car class represents a car in the game, with methods for movement, collision detection, and rendering.
class Car:
    def __init__(self, x, y):
        self.pt = myPoint(x, y)
        self.x = x
        self.y = y
        self.width = 14
        self.height = 30
        self.points = 0

        # Car image loading and initialization
        self.original_image = pygame.image.load("C:\\Users\\yashr\\Desktop\\DDQN-Car-Racing\\car.png").convert()
        self.image = self.original_image  # This will reference the rotated image.
        self.image.set_colorkey((0,0,0))
        self.rect = self.image.get_rect().move(self.x, self.y)
        self.angle = math.radians(180)
        self.soll_angle = self.angle

        ## Car velocity and acceleration initialization
        self.dvel = 1
        self.vel = 0
        self.velX = 0
        self.velY = 0
        self.maxvel = 13 # before 15
        self.angle = math.radians(180)
        self.soll_angle = self.angle

        # Car points initialization
        self.pt1 = myPoint(self.pt.x - self.width / 2, self.pt.y - self.height / 2)
        self.pt2 = myPoint(self.pt.x + self.width / 2, self.pt.y - self.height / 2)
        self.pt3 = myPoint(self.pt.x + self.width / 2, self.pt.y + self.height / 2)
        self.pt4 = myPoint(self.pt.x - self.width / 2, self.pt.y + self.height / 2)

        self.p1 = self.pt1
        self.p2 = self.pt2
        self.p3 = self.pt3
        self.p4 = self.pt4

        self.distances = []
    
    # Action method to control the car's movement based on the given choice.
    def action(self, choice):
        if choice == 0:
            pass
        elif choice == 1:
            self.accelerate(self.dvel)
        elif choice == 2:
            self.turn(-1)
        elif choice == 3:
            self.turn(1)
        elif choice == 4:
            self.accelerate(-self.dvel)

        # Additional actions for complex maneuvering (not used in the current implementation)
        """
        elif choice == 5:
            self.accelerate(-self.dvel)
            self.turn(1)
        elif choice == 6:
            self.accelerate(-self.dvel)
            self.turn(-1)
        elif choice == 7:
            self.accelerate(self.dvel)
            self.turn(-1)
        elif choice == 8:
            self.accelerate(self.dvel)
            self.turn(1)
        """

        pass
    
    # Accelerate method to increase or decrease the car's velocity.
    def accelerate(self,dvel):
        dvel = dvel * 2
        self.vel = self.vel + dvel

        if self.vel > self.maxvel:
            self.vel = self.maxvel
        
        if self.vel < -self.maxvel:
            self.vel = -self.maxvel
        
    # Turn method to change the car's angle based on the direction.
    def turn(self, dir):
        self.soll_angle = self.soll_angle + dir * math.radians(15)
    
    # Update method to move the car based on its velocity and angle.
    def update(self):
        # Drifting code
        if(self.soll_angle > self.angle):
            if(self.soll_angle > self.angle + math.radians(10) * self.maxvel / ((self.velX**2 + self.velY**2)**0.5 + 1)):
                self.angle = self.angle + math.radians(10) * self.maxvel / ((self.velX**2 + self.velY**2)**0.5 + 1)
            else:
                self.angle = self.soll_angle
        
        if(self.soll_angle < self.angle):
            if(self.soll_angle < self.angle - math.radians(10) * self.maxvel / ((self.velX**2 + self.velY**2)**0.5 + 1)):
                self.angle = self.angle - math.radians(10) * self.maxvel / ((self.velX**2 + self.velY**2)**0.5 + 1)
            else:
                self.angle = self.soll_angle

        # Car movement code
        self.angle = self.soll_angle
        vec_temp = rotate(myPoint(0,0), myPoint(0,self.vel), self.angle)
        self.velX, self.velY = vec_temp.x, vec_temp.y
        self.x = self.x + self.velX
        self.y = self.y + self.velY
        self.rect.center = self.x, self.y

        self.pt1 = myPoint(self.pt1.x + self.velX, self.pt1.y + self.velY)
        self.pt2 = myPoint(self.pt2.x + self.velX, self.pt2.y + self.velY)
        self.pt3 = myPoint(self.pt3.x + self.velX, self.pt3.y + self.velY)
        self.pt4 = myPoint(self.pt4.x + self.velX, self.pt4.y + self.velY)

        self.p1 ,self.p2 ,self.p3 ,self.p4  = rotateRect(self.pt1, self.pt2, self.pt3, self.pt4, self.soll_angle)

        # Update the car's image based on its angle
        self.image = pygame.transform.rotate(self.original_image, 90 - self.soll_angle * 180 / math.pi)
        x, y = self.rect.center  # Save its current center.
        self.rect = self.image.get_rect()  # Replace old rect with new rect.
        self.rect.center = (x, y)

    # Cast method to create rays from the car's position and check for intersections with walls.
    def cast(self, walls):
        ray1 = Ray(self.x, self.y, self.soll_angle)
        ray2 = Ray(self.x, self.y, self.soll_angle - math.radians(30))
        ray3 = Ray(self.x, self.y, self.soll_angle + math.radians(30))
        ray4 = Ray(self.x, self.y, self.soll_angle + math.radians(45))
        ray5 = Ray(self.x, self.y, self.soll_angle - math.radians(45))
        ray6 = Ray(self.x, self.y, self.soll_angle + math.radians(90))
        ray7 = Ray(self.x, self.y, self.soll_angle - math.radians(90))
        ray8 = Ray(self.x, self.y, self.soll_angle + math.radians(180))
        ray9 = Ray(self.x, self.y, self.soll_angle + math.radians(10))
        ray10 = Ray(self.x, self.y, self.soll_angle - math.radians(10))
        ray11 = Ray(self.x, self.y, self.soll_angle + math.radians(135))
        ray12 = Ray(self.x, self.y, self.soll_angle - math.radians(135))
        ray13 = Ray(self.x, self.y, self.soll_angle + math.radians(20))
        ray14 = Ray(self.x, self.y, self.soll_angle - math.radians(20))
        ray15 = Ray(self.p1.x,self.p1.y, self.soll_angle + math.radians(90))
        ray16 = Ray(self.p2.x,self.p2.y, self.soll_angle - math.radians(90))
        ray17 = Ray(self.p1.x,self.p1.y, self.soll_angle + math.radians(0))
        ray18 = Ray(self.p2.x,self.p2.y, self.soll_angle - math.radians(0))

        self.rays = []
        self.rays.append(ray1)
        self.rays.append(ray2)
        self.rays.append(ray3)
        self.rays.append(ray4)
        self.rays.append(ray5)
        self.rays.append(ray6)
        self.rays.append(ray7)
        self.rays.append(ray8)
        self.rays.append(ray9)
        self.rays.append(ray10)
        self.rays.append(ray11)
        self.rays.append(ray12)
        self.rays.append(ray13)
        self.rays.append(ray14)
        self.rays.append(ray15)
        self.rays.append(ray16)
        self.rays.append(ray17)
        self.rays.append(ray18)

        observations = []
        self.closestRays = []

        for ray in self.rays:
            closest = None #myPoint(0,0)
            record = math.inf
            for wall in walls:
                pt = ray.cast(wall)
                if pt:
                    dist = distance(myPoint(self.x, self.y),pt)
                    if dist < record:
                        record = dist
                        closest = pt

            if closest: 
                self.closestRays.append(closest)
                observations.append(record) 
            else:
                observations.append(1000)

        for i in range(len(observations)):
            #invert observation values 0 is far away 1 is close
            observations[i] = ((1000 - observations[i]) / 1000)

        observations.append(self.vel / self.maxvel)
        return observations

    # Collision detection method to check if the car collides with a wall.
    def collision(self, wall):
        line1 = myLine(self.p1, self.p2)
        line2 = myLine(self.p2, self.p3)
        line3 = myLine(self.p3, self.p4)
        line4 = myLine(self.p4, self.p1)

        x1 = wall.x1 
        y1 = wall.y1
        x2 = wall.x2
        y2 = wall.y2

        lines = []
        lines.append(line1)
        lines.append(line2)
        lines.append(line3)
        lines.append(line4)

        for li in lines: 
            x3 = li.pt1.x
            y3 = li.pt1.y
            x4 = li.pt2.x
            y4 = li.pt2.y

            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            
            if(den == 0):
                den = 0
            else:
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

                if t > 0 and t < 1 and u < 1 and u > 0:
                    return(True)
        
        return(False)
    
    # Score method to check if the car scores by passing through a goal.
    def score(self, goal):
        vec = rotate(myPoint(0,0), myPoint(0,-50), self.angle)
        line1 = myLine(myPoint(self.x,self.y),myPoint(self.x + vec.x, self.y + vec.y))

        x1 = goal.x1 
        y1 = goal.y1
        x2 = goal.x2
        y2 = goal.y2
            
        x3 = line1.pt1.x
        y3 = line1.pt1.y
        x4 = line1.pt2.x
        y4 = line1.pt2.y

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if(den == 0):
            den = 0
        else:
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

            if t > 0 and t < 1 and u < 1 and u > 0:
                pt = math.floor(x1 + t * (x2 - x1)), math.floor(y1 + t * (y2 - y1))
                d = distance(myPoint(self.x, self.y), myPoint(pt[0], pt[1]))
                if d < 20:
                    #pygame.draw.circle(win, (0,255,0), pt, 5)
                    self.points += GOALREWARD
                    return(True)

        return(False)

    # Reset method to reset the car's position and state.
    def reset(self):
        self.x = 150
        self.y = 75
        self.velX = 0
        self.velY = 0
        self.vel = 0
        self.angle = math.radians(180)
        self.soll_angle = self.angle
        self.points = 0

        self.pt1 = myPoint(self.pt.x - self.width / 2, self.pt.y - self.height / 2)
        self.pt2 = myPoint(self.pt.x + self.width / 2, self.pt.y - self.height / 2)
        self.pt3 = myPoint(self.pt.x + self.width / 2, self.pt.y + self.height / 2)
        self.pt4 = myPoint(self.pt.x - self.width / 2, self.pt.y + self.height / 2)

        self.p1 = self.pt1
        self.p2 = self.pt2
        self.p3 = self.pt3
        self.p4 = self.pt4

    # Draw method to render the car on the screen.
    def draw(self, win):
        win.blit(self.image, self.rect)
  
# The RacingEnv class represents the racing environment, including the car, walls, and goals.
class RacingEnv:
    def __init__(self):
        pygame.init()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 36)
        self.fps = 120
        self.width = 1000
        self.height = 600
        self.history = []

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RACING DQN")
        self.screen.fill((0,0,0))
        self.back_image = pygame.image.load("C:\\Users\\yashr\\Desktop\\DDQN-Car-Racing\\track_new.png").convert()
        self.back_rect = self.back_image.get_rect().move(0, 0)
        self.action_space = None
        self.observation_space = None
        self.game_reward = 0
        self.score = 0

        self.reset()

    # Reset method to initialize the game state, including the car, walls, and goals.
    def reset(self):
        self.screen.fill((0, 0, 0))
        self.car = Car(150, 75)
        self.walls = getWalls()
        self.goals = getGoals()
        self.game_reward = 0

    # Step method to perform an action in the environment and return the new state, reward, and done flag.
    def step(self, action):
        done = False
        self.car.action(action)
        self.car.update()
        reward = LIFE_REWARD

        # Check if car passes Goal and scores
        index = 1
        for goal in self.goals:
            if index > len(self.goals):
                index = 1
            if goal.isactiv:
                if self.car.score(goal):
                    goal.isactiv = False
                    self.goals[index-2].isactiv = True
                    reward += GOALREWARD
            index = index + 1

        # Check if car collides with any wall
        for wall in self.walls:
            if self.car.collision(wall):
                reward += PENALTY
                done = True

        new_state = self.car.cast(self.walls)

        if done:
            new_state = None

        return new_state, reward, done

    # Render method to display the current state of the game, including the car, walls, and goals.
    def render(self, action):
        DRAW_WALLS = False
        DRAW_GOALS = True
        DRAW_RAYS = False

        pygame.time.delay(10)

        self.clock = pygame.time.Clock()
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.back_image, self.back_rect)

        # Draw the walls, goals and rays if enabled
        if DRAW_WALLS:
            for wall in self.walls:
                wall.draw(self.screen)
        
        if DRAW_GOALS:
            for goal in self.goals:
                goal.draw(self.screen)
                if goal.isactiv:
                    goal.draw(self.screen)
        
        if DRAW_RAYS:
            i = 0
            for pt in self.car.closestRays:
                pygame.draw.circle(self.screen, (0,0,255), (pt.x, pt.y), 5)
                i += 1
                if i < 15:
                    pygame.draw.line(self.screen, (255,255,255), (self.car.x, self.car.y), (pt.x, pt.y), 1)
                elif i >=15 and i < 17:
                    pygame.draw.line(self.screen, (255,255,255), ((self.car.p1.x + self.car.p2.x)/2, (self.car.p1.y + self.car.p2.y)/2), (pt.x, pt.y), 1)
                elif i == 17:
                    pygame.draw.line(self.screen, (255,255,255), (self.car.p1.x , self.car.p1.y ), (pt.x, pt.y), 1)
                else:
                    pygame.draw.line(self.screen, (255,255,255), (self.car.p2.x, self.car.p2.y), (pt.x, pt.y), 1)

        # Draw the car
        self.car.draw(self.screen)
        
        # Draw the action space
        pygame.draw.rect(self.screen,(255,255,255),(800, 100, 40, 40),2)
        pygame.draw.rect(self.screen,(255,255,255),(850, 100, 40, 40),2)
        pygame.draw.rect(self.screen,(255,255,255),(900, 100, 40, 40),2)
        pygame.draw.rect(self.screen,(255,255,255),(850, 50, 40, 40),2)

        if action == 1:
            pygame.draw.rect(self.screen,(0,255,0),(850, 100, 40, 40)) 
        elif action == 2:
            pygame.draw.rect(self.screen,(0,255,0),(800, 100, 40, 40))
        elif action == 3:
            pygame.draw.rect(self.screen,(0,255,0),(900, 100, 40, 40))
        elif action == 4:
            pygame.draw.rect(self.screen,(0,255,0),(850, 50, 40, 40)) 

        # Additional actions for complex maneuvering (not used in the current implementation)
        """
        elif action == 5:
            pygame.draw.rect(self.screen,(0,255,0),(850, 50, 40, 40))
            pygame.draw.rect(self.screen,(0,255,0),(900, 100, 40, 40))
        elif action == 6:
            pygame.draw.rect(self.screen,(0,255,0),(850, 50, 40, 40))
            pygame.draw.rect(self.screen,(0,255,0),(800, 100, 40, 40))
        elif action == 7:
            pygame.draw.rect(self.screen,(0,255,0),(850, 100, 40, 40))
            pygame.draw.rect(self.screen,(0,255,0),(900, 100, 40, 40))
        elif action == 8:
            pygame.draw.rect(self.screen,(0,255,0),(850, 100, 40, 40))
            pygame.draw.rect(self.screen,(0,255,0),(800, 100, 40, 40))
        """

        # Score display
        text_surface = self.font.render(f'Points {self.car.points}', True, pygame.Color('green'))
        self.screen.blit(text_surface, dest=(0, 0))
        # Speed display
        text_surface = self.font.render(f'Speed {self.car.vel*-1}', True, pygame.Color('green'))
        self.screen.blit(text_surface, dest=(800, 0))

        self.clock.tick(self.fps)
        pygame.display.update()

    # Quit method to close the game window and clean up resources.
    def close(self):
        pygame.quit()