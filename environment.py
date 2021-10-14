import pygame
import random
import math
import numpy as np
from collections import namedtuple

pygame.init()

Size = namedtuple('Size', 'w h')

# Window
WINDOW = Size(400, 1000)
WINDOW_RECT = pygame.Rect(0, 0, WINDOW.w, WINDOW.h)
INDENT = 15

# Game
SPEED = 1000
BLACK = (0, 0, 0)


class CapsuleLander:
    __size = Size(70, 70)
    __img = pygame.image.load('capsule.png')
    __engine_power = 1
    __side_engine_power = 2
    __gravity = 2

    def __init__(self):
        # Init render stuff
        self.display = pygame.display.set_mode(WINDOW)
        pygame.display.set_caption('Space Landing')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font('arial.ttf', 25)

        # Capsule
        self.img = pygame.transform.scale(self.__img, self.__size)
        self.x = random.randint(self.__size.w + INDENT, WINDOW.w - self.__size.w - INDENT)
        self.y = self.__size.h + INDENT
        self.angle = random.randint(0, 359)
        self.distance = 0.0
        self.falling_speed = 0
        self.state = np.array([self.x,
                               self.y,
                               self.angle,
                               self.distance,
                               self.falling_speed])
        self.capsule_rect = self.img.get_rect(center=(self.x, self.y))

        # World
        self.surface_rect = pygame.Rect(0, WINDOW.h - 100, WINDOW.w, WINDOW.h)
        self.action = np.zeros(4)

    def step(self, action):
        """Play game step"""
        assert 0 <= action < self.action.size

        # For closing window without crushing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Update capsule rect
        self.capsule_rect = self.img.get_rect(center=(self.x, self.y))

        # Apply action
        prev_angle = self.angle
        prev_y = self.y
        if action == 0:
            self.__set_angle(self.__side_engine_power)
            self.falling_speed = self.__gravity
        elif action == 1:
            self.x, self.y = calculate_new_pos(self.x, self.y, self.__engine_power, self.angle)
            self.falling_speed = self.__gravity - (prev_y - self.y)
        elif action == 2:
            self.__set_angle(-self.__side_engine_power)
            self.falling_speed = self.__gravity
        else:
            self.falling_speed = self.__gravity

        # Calculate y position and distance to the ground
        self.y += self.__gravity
        self.distance = self.surface_rect.y - self.y

        # Draw game
        self.render()

        # Calculate reward
        if self.distance < 200:  # with distance 200 capsule will have time to land correctly
            reward = self.__get_reward(prev_angle)
            reward += 2 if self.__get_score() else -2
        else:
            reward = -self.__get_reward(prev_angle)
            if self.falling_speed > self.__gravity:
                reward += self.falling_speed - self.__gravity

        # Check if is terminal
        is_terminal = False
        if self.__is_colliding(self.surface_rect):
            is_terminal = True
            score = self.__get_score()
            reward = score if score else -100
        if self.__is_out_of_bounds():
            is_terminal = True
            reward = -100

        return self.__get_norm_state(), reward, is_terminal, {}

    def __set_angle(self, val):
        """Set angle and save values from 0 to 359"""
        self.angle += val
        if self.angle > 359:
            self.angle = -1 + self.angle - 359
        if self.angle < 0:
            self.angle = 359 + self.angle

    def __is_colliding(self, rect):
        return self.capsule_rect.colliderect(rect)

    def __is_out_of_bounds(self):
        return not self.capsule_rect.colliderect(WINDOW_RECT)

    def __get_score(self):
        """Get score based on capsule angle where 90 is North"""
        if 40 <= self.angle <= 140:
            if self.angle <= 90:
                return 100 * (self.angle / 90)
            else:
                return 100 * (1 - (self.angle - 90) / 90)
        else:
            return 0

    def __get_reward(self, previous_angle):
        """Get reward based on capsule angle where 90 is North"""
        if self.angle != previous_angle:  # check if capsule got rotated
            if 90 < self.angle <= 270:  # if capsule nose points left
                return 1 if previous_angle > self.angle else -1  # we will reward going clockwise
            else:  # if our nose points right
                if abs(self.angle - previous_angle) > self.__side_engine_power:  # step through 359 0 threshold
                    return 1 if previous_angle > self.angle else -1  # reward going counterclockwise
                return 1 if previous_angle < self.angle else -1  # reward going counterclockwise
        return 0

    def __get_norm_state(self):
        """Normalize state values in range -1 to 1"""
        range = [-1, 1]
        n_state = [np.interp(self.x, [0 - self.__size.w, WINDOW.w + self.__size.w], range),
                   np.interp(self.y, [0, self.surface_rect.y], range),
                   np.interp(self.angle, [0, 359], range),
                   np.interp(self.distance, [self.__size.h / 2, self.surface_rect.y], range),
                   np.interp(self.falling_speed, [self.__gravity - self.__engine_power,
                                                  self.__gravity + self.__engine_power], range)]
        return np.array(n_state, dtype=float)

    def get_random_action(self):
        return random.randint(0, self.action.size - 1)

    def reset(self):
        """Reset the game"""
        self.x = random.randint(self.__size.w + INDENT, WINDOW.w - self.__size.w - INDENT)
        self.y = self.__size.h + INDENT
        self.angle = random.randint(0, 359)
        self.distance = 0.0
        self.falling_speed = 0.0
        self.state = np.array([self.x,
                               self.y,
                               self.angle,
                               self.distance,
                               self.falling_speed])
        self.capsule_rect = self.img.get_rect(center=(self.x, self.y))
        return self.step(3)[0]

    def render(self):
        # Background
        self.display.fill((255, 255, 255))

        # Surface
        pygame.draw.rect(self.display, BLACK, self.surface_rect)

        # Capsule
        r_img, r_rect = rotate(self.img,
                               self.angle - 90,
                               (self.x, self.y))
        self.display.blit(r_img, r_rect)

        # Statistics
        self.__draw_statistics()

        # Pygame
        pygame.display.flip()
        self.clock.tick(SPEED)

    def __draw_statistics(self):
        font_h = self.font.size('S')[1]  # font height
        self.display.blit(self.font.render('Angle: {:.2f}'.format(self.angle), True, BLACK),
                          (0, 0 + font_h * 0))
        self.display.blit(self.font.render('Distance: {:.2f}'.format(self.distance), True, BLACK),
                          (0, 0 + font_h * 1))
        self.display.blit(self.font.render('Falling Speed: {:.2f}'.format(self.falling_speed), True, BLACK),
                          (0, 0 + font_h * 2))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def rotate(surface, angle, pos):
    r_surface = pygame.transform.rotozoom(surface, angle, 1)
    r_rect = r_surface.get_rect(center=pos)
    return r_surface, r_rect


# ------------------------------------------------------------------
def calculate_new_pos(x, y, speed, angle):
    angle_in_radians = angle * math.pi / 180
    new_x = x + speed * math.cos(angle_in_radians)
    new_y = y - speed * math.sin(angle_in_radians)
    return new_x, new_y
