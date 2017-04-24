#!/usr/bin/env python3

# Cheng Hann Gan, James Park
# Must install pyglet (https://bitbucket.org/pyglet/pyglet/wiki/Home) for audio.

import asyncio
import math
import sys

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


async def hitBall(robot):
    """Makes Cozmo hit the ball.

    Args:
        robot: The object representing Cozmo.
    """
    robot.stop_all_motors()
    distance = 2 * robot.BACK_DISTANCE
    await doActionWithTimeout(robot.drive_straight(distance_mm(distance), speed_mmps(robot.HIT_SPEED)), 5)
    robot.add_odom_forward(robot, distance)


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
            await robot.drive_wheels(0, 0, robot.ROBOT_ACCELERATION, robot.ROBOT_ACCELERATION)
            return planning.PathPlan()
        self.move_timer -= 1
        if self.move_timer <= 0:
            distance = robot.BACK_DISTANCE
            await doActionWithTimeout(robot.drive_straight(distance_mm(distance), speed_mmps(robot.HIT_SPEED)), 5)
            robot.add_odom_forward(robot, distance)
        elif self.turn_timer > 0:
            self.turn_timer -= 1
        else:
            await doActionWithTimeout(robot.turn_in_place(degrees(robot.SEARCH_TURN), num_retries = 3), 0.5)
            robot.add_odom_rotation(robot, robot.SEARCH_TURN)
            self.turn_timer = robot.TURN_COUNTER

class HitBall(State):
    """Moves Cozmo's arm to hit the ball."""

    async def update(self, robot):
        """
        Executes the state's behavior for the current tick.

        Args:
            robot: The object to affect behavior for.
        
        Returns:
            If the object's state should be changed, returns the class of the new state.
            Otherwise, return None.
        """
        rotation_rad = math.radians(robot.rotation)
        turn = planning.getTurnDirection(math.cos(rotation_rad), math.sin(rotation_rad), robot.position, robot.ball)
        robot.stop_all_motors()
        await robot.turn_in_place(radians(turn), num_retries = 3).wait_for_completed()
        robot.add_odom_rotation(robot, math.degrees(turn))

        robot.stop_all_motors()
        await hitBall(robot)
        robot.ball = None
        robot.ball_grid = None
        return Search()


if __name__ == '__main__':
    cozmo.run_program(run, use_viewer=True, force_viewer_on_top=True)
