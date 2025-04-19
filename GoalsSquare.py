"""
Name: Priyanshu Ranka (NUID: 002305396)
Project: DDQN for Car Racing
Course: CS 5180 - Reinforcement Learning
Professor: Prof. Robert Platt
Semester: Spring 2025
Description: This file contains the Goals for square track.
"""

import pygame
import math

# Goal class to represent the goal lines
class Goal:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.isactiv = False
    
    # Draw the goal line and active goal
    def draw(self, win):
        # Draw the goal line in green
        pygame.draw.line(win, (0, 255, 0), (self.x1, self.y1), (self.x2, self.y2), 2)
        if self.isactiv:
            # If the goal is active, draw it in red
            pygame.draw.line(win, (255, 0, 0), (self.x1, self.y1), (self.x2, self.y2), 2)

# Wall class to represent the walls of the track
class Wall:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def draw(self, win):
        # Draw the wall in white
        pygame.draw.line(win, (255, 255, 255), (self.x1, self.y1), (self.x2, self.y2), 5)

# Function to get the walls of the track
def getWalls():
    walls = []

    # Define the walls of the track (sample walls)
    wall1 = Wall(100, 50, 900, 50)
    wall2 = Wall(900, 50, 900, 550)
    wall3 = Wall(900, 550, 100, 550)
    wall4 = Wall(100, 550, 100, 50)

    # Add more walls as needed for your track
    walls.append(wall1)
    walls.append(wall2)
    walls.append(wall3)
    walls.append(wall4)

    return walls

# Function to get the goals
def getGoals():
    goals = []

    # The center of the circle (500, 300)
    center_x, center_y = 500, 300

    # Define the walls to which the rays will be cast (we will use the existing `getWalls()` function)
    walls = getWalls()

    # Calculate the number of rays (for example, 25 rays, evenly spaced)
    num_rays = 20
    angle_step = 360 / num_rays

    for i in range(num_rays):
        angle = math.radians(i * angle_step + 198)  # Angle for each ray

        # Cast the ray out from the center of the circle at the calculated angle
        ray_end_x = center_x + math.cos(angle) * 1000  # Arbitrary length of the ray
        ray_end_y = center_y + math.sin(angle) * 1000

        # Check where the ray intersects with each wall
        for wall in walls:
            intersection = get_intersection(center_x, center_y, ray_end_x, ray_end_y, wall)
            if intersection:
                goal = Goal(center_x, center_y, intersection[0], intersection[1])  # Create the goal at the intersection
                goals.append(goal)

    # Set the first goal as active
    if goals:
        goals[0].isactiv = True

    return goals

# Function to check if two line segments intersect
def get_intersection(x1, y1, x2, y2, wall):
    # Line segment 1 (the ray)
    x3, y3 = x1, y1
    x4, y4 = x2, y2

    # Line segment 2 (the wall)
    x5, y5 = wall.x1, wall.y1
    x6, y6 = wall.x2, wall.y2

    # Denominator of the intersection calculation
    den = (x1 - x2) * (y5 - y6) - (y1 - y2) * (x5 - x6)

    if den == 0:
        return None  # No intersection (parallel lines)

    # Calculate the intersection point using the determinant method
    t = ((x3 - x5) * (y5 - y6) - (y3 - y5) * (x5 - x6)) / den
    u = -((x3 - x4) * (y3 - y5) - (y3 - y4) * (x3 - x5)) / den

    if 0 <= t <= 1 and 0 <= u <= 1:
        # If both t and u are between 0 and 1, the lines intersect
        ix = x3 + t * (x4 - x3)
        iy = y3 + t * (y4 - y3)
        return (ix, iy)

    return None  # No valid intersection

# Function to render the goals and rays
def render_goals_and_rays(win, goals, walls, center_x, center_y):
    # Draw all the walls
    for wall in walls:
        wall.draw(win)

    # Draw rays from the center of the circle
    for i in range(len(goals)):
        goal = goals[i]
        # Draw the ray from the center of the circle to the goal's start point
        pygame.draw.line(win, (255, 255, 255), (center_x, center_y), (goal.x1, goal.y1), 2)
        goal.draw(win)  # Draw the goal line

# Main function to run the visualization
def main():
    pygame.init()

    # Create the display window
    WIDTH, HEIGHT = 1000, 600
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Wall and Goal Visualization")

    # Get the walls and goals
    walls = getWalls()
    goals = getGoals()

    # Main loop
    clock = pygame.time.Clock()
    running = True
    while running:
        clock.tick(60)  # Limit to 60 FPS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        win.fill((0, 0, 0))  # Black background

        # Draw goals and rays
        render_goals_and_rays(win, goals, walls, 500, 300)  # center of the circle is (500, 300)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()