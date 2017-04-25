#!/usr/bin/env python3

# Cheng Hann Gan, James Park
# Must install pyglet (https://bitbucket.org/pyglet/pyglet/wiki/Home) for audio.

import asyncio
import math
import sys
import time

import cozmo
import cv2
import find_ball, find_goal
import numpy as np
from cozmo.util import Angle, degrees, radians, distance_mm, speed_mmps

import planning
import play_soccer

try:
    from PIL import ImageDraw, ImageFont
except ImportError:
    sys.exit('run `pip3 install --user Pillow numpy` to run this example.')

from state_machine import State, StateMachine

async def doActionWithTimeout(action, timeout):
    """Executes an action, canceling it after a certain amount of time if it doesn't finish.

    Args:
        action: The action to execute.
        timeout: The amount of time (in seconds) that the action can be executed for before being cancelled.
    """
    try:
        await action.wait_for_completed(timeout)
    except asyncio.TimeoutError:
        if action.is_running:
            try:
                action.abort()
            except KeyError:
                pass


async def hitBall(robot, distance_multiplier):
    """Makes Cozmo hit the ball.

    Args:
        robot: The object representing Cozmo.
        distance_multiplier: Multiplier for the distance that the robot will travel to hit the ball.
    """
    robot.stop_all_motors()
    distance = 2 * robot.BACK_DISTANCE * distance_multiplier
    timeout = 3
    await doActionWithTimeout(robot.drive_straight(distance_mm(distance), speed_mmps(robot.HIT_SPEED), should_play_anim = False), timeout)
    robot.stop_all_motors()
    await doActionWithTimeout(robot.drive_straight(distance_mm(-distance), speed_mmps(robot.HIT_SPEED), should_play_anim = False), timeout)


class StateTemplate(State):
    """
    """

    async def update(self, robot):
        """Executes the state's behavior for the current tick.

        Args:
            robot: The object to affect behavior for.
        
        Returns:
            If the object's state should be changed, returns the class of the new state.
            Otherwise, return None.
        """
        pass


class Search(State):
    """Searches for the ball.
    """

    """The direction that the robot will turn in."""
    _TURN_DIRECTION = 1

    def __init__(self, args=None):
        """Initializes the state when it is first switched to.

        Args:
            args: Whether the robot will spin counterclockwise.
        """
        if args:
            self._TURN_DIRECTION = -1
        self.turning = False
        self.turn_timer = 0
        self.first = True
        self.found_ball = False
        self.looked_ball = False
        self.goal_look_timer = 5
        self.ball_look_timer = 5

    async def update(self, robot):
        """Executes the state's behavior for the current tick.

        Args:
            robot: The object to affect behavior for.
        
        Returns:
            If the object's state should be changed, returns the class of the new state.
            Otherwise, return None.
        """
        if self.first or self.move_timer <= 0:
            self.turn_timer = robot.TURN_COUNTER
            self.move_timer = 360 * 2.25 / robot.TURN_COUNTER
            self.first = False
        if robot.localized and robot.ball is not None:
            if not self.found_ball:
                rotation_rad = math.radians(robot.rotation)
                turn_to_goal = planning.getTurnDirection(math.cos(rotation_rad), math.sin(rotation_rad), robot.position, robot.goal_position)
                await robot.turn_in_place(radians(turn_to_goal), num_retries = 3).wait_for_completed()
                robot.stop_all_motors()
                robot.add_odom_rotation(robot, math.degrees(turn_to_goal))
                self.found_ball = True
            if self.goal_look_timer > 0:
                self.goal_look_timer -= 1
            elif not self.looked_ball:
                self.looked_ball = True
                rotation_rad = math.radians(robot.rotation)
                turn = planning.getTurnDirection(math.cos(rotation_rad), math.sin(rotation_rad), robot.position, robot.ball)
                robot.stop_all_motors()
                robot.ball = None
                robot.ball_grid = None
                await robot.turn_in_place(radians(turn), num_retries = 3).wait_for_completed()
                robot.add_odom_rotation(robot, math.degrees(turn))
            else:
                self.ball_look_timer -= 1
            if self.ball_look_timer <= 0:
                if robot.ball is not None:
                    return planning.PathPlan()
                else:
                    return Search()
            return
        self.move_timer -= 1
        if self.move_timer <= 0:
            distance = robot.BACK_DISTANCE
            await doActionWithTimeout(robot.drive_straight(distance_mm(distance), speed_mmps(robot.HIT_SPEED), should_play_anim = False), 5)
            robot.add_odom_forward(robot, distance)
        elif self.turn_timer > 0:
            self.turn_timer -= 1
        else:
            await doActionWithTimeout(robot.turn_in_place(degrees(robot.SEARCH_TURN), num_retries = 3), 0.5)
            robot.stop_all_motors()
            robot.add_odom_rotation(robot, robot.SEARCH_TURN * .8)
            self.turn_timer = robot.TURN_COUNTER

class HitBall(State):
    """Moves Cozmo into the ball."""

    # The time that Cozmo will wait before hitting the ball.
    WAIT_TIME = 5

    def __init__(self, args=None):
        """Initializes the state when it is first switched to.

        Args:
            args: Multiplier for how far the robot will move to hit the ball.
        """
        if args:
            self.distance_multiplier = args
            self.turned = True
        else:
            self.distance_multiplier = 1
            self.turned = False
        self.wait = HitBall.WAIT_TIME
        self.hit = False

    async def update(self, robot):
        """
        Executes the state's behavior for the current tick.

        Args:
            robot: The object to affect behavior for.
        
        Returns:
            If the object's state should be changed, returns the class of the new state.
            Otherwise, return None.
        """
        if not self.turned:
            rotation_rad = math.radians(robot.rotation)
            turn = planning.getTurnDirection(math.cos(rotation_rad), math.sin(rotation_rad), robot.position, robot.ball)
            robot.stop_all_motors()
            await robot.turn_in_place(radians(turn), num_retries = 3).wait_for_completed()
            robot.add_odom_rotation(robot, math.degrees(turn))
            robot.ball = None
            robot.ball_grid = None
            self.turned = True
        elif not self.hit and (robot.ball is not None or self.wait <= 0):
            robot.stop_all_motors()
            await hitBall(robot, self.distance_multiplier)
            robot.ball = None
            robot.ball_grid = None
            robot.localized = False
            self.hit = True
        elif self.wait <= -HitBall.WAIT_TIME:
            return Search()
        self.wait -= 1


if __name__ == '__main__':
    cozmo.run_program(run, use_viewer=True, force_viewer_on_top=True)
