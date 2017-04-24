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
from cozmo.util import Angle, degrees, distance_mm, speed_mmps

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
    await doActionWithTimeout(robot.drive_straight(distance_mm(2 * robot.BACK_DISTANCE), speed_mmps(robot.HIT_SPEED)), 5)


class StateTemplate(State):
    """
    """

    async def update(self, owner):
        """Executes the state's behavior for the current tick.

        Args:
            owner: The object to affect behavior for.
        
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
        self.counter = 0
        self.first = True

    async def update(self, owner):
        """Executes the state's behavior for the current tick.

        Args:
            owner: The object to affect behavior for.
        
        Returns:
            If the object's state should be changed, returns the class of the new state.
            Otherwise, return None.
        """
        if self.first:
            self.counter = owner.TURN_COUNTER
            self.first = False
            return
        if self.turning:
            owner.add_odom_rotation(owner, owner.TURN_YAW * owner.delta_time * self._TURN_DIRECTION * -1)
        if owner.localized and owner.ball is not None:
            await owner.drive_wheels(0, 0, owner.ROBOT_ACCELERATION, owner.ROBOT_ACCELERATION)
            return planning.PathPlan()
        if self.counter <= 0:
            self.counter = 
        turn_speed = owner.TURN_SPEED * self._TURN_DIRECTION
        await owner.drive_wheels(turn_speed, -turn_speed, owner.ROBOT_ACCELERATION, owner.ROBOT_ACCELERATION)
        self.turning = True

class HitBall(State):
    """Moves Cozmo's arm to hit the ball."""

    async def update(self, owner):
        """
        Executes the state's behavior for the current tick.

        Args:
            owner: The object to affect behavior for.
        
        Returns:
            If the object's state should be changed, returns the class of the new state.
            Otherwise, return None.
        """
        owner.stop_all_motors()
        await hitBall(owner)
        owner.ball = None
        owner.ball_grid = None
        return Search()


if __name__ == '__main__':
    cozmo.run_program(run, use_viewer=True, force_viewer_on_top=True)
