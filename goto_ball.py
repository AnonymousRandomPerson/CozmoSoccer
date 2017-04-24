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
    back_distance = robot._PHYSICAL_RADIUS * 20
    await doActionWithTimeout(owner.drive_straight(distance_mm(back_distance), speed_mmps(1000), in_parallel = True), 1)
    await doActionWithTimeout(owner.drive_straight(distance_mm(1.5 * back_distance), speed_mmps(1000), in_parallel = True), 1)
    # await robot.set_lift_height(0, 100, 100, in_parallel = True).wait_for_completed()
    # await robot.set_lift_height(1, 100, 100, in_parallel = True).wait_for_completed()


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

    async def update(self, owner):
        """Executes the state's behavior for the current tick.

        Args:
            owner: The object to affect behavior for.
        
        Returns:
            If the object's state should be changed, returns the class of the new state.
            Otherwise, return None.
        """
        if self.turning:
            owner.add_odom_rotation(owner, owner.TURN_YAW * owner.delta_time * self._TURN_DIRECTION * -1)
        if owner.ball:
            await owner.drive_wheels(0, 0, owner.ROBOT_ACCELERATION, owner.ROBOT_ACCELERATION)
            return planning.PathPlan()
        turn_speed = owner.TURN_SPEED * self._TURN_DIRECTION
        await owner.drive_wheels(turn_speed, -turn_speed, owner.ROBOT_ACCELERATION, owner.ROBOT_ACCELERATION)
        self.turning = True


class Approach(State):
    """Approaches a ball that has been seen.
    """

    """The number of frames after seeing no ball to start rotating again."""
    _BALL_TIMEOUT = 5

    """Angle threshold for the robot to start approaching the ball."""
    _ROTATE_THRESHOLD = 10
    """The radius of the physical ball in cm."""
    _PHYSICAL_RADIUS = 2
    """The horizontal angle of view of the robot's camera (in degrees)."""
    _ANGLE_OF_VIEW = 60
    """Tangent of half of the field of view, used to figure out the distance away from the ball."""
    _TAN_ANGLE = math.tan(math.radians(_ANGLE_OF_VIEW) / 2)
    """The distance between the robot's camera and the front of its wheels (in cm)."""
    _COZMO_OFFSET = 1.2

    """The size (in pixels) of the ball that will cause the robot to back up after losing sight of it."""
    _BALL_BACK_UP = 100

    """The maximum movement speed of the robot."""
    _MOVE_SPEED = 500

    """The minimum speed multiplier of the robot when close to the ball."""
    _MIN_SPEED_MULTIPLIER = 0.25

    """The distance (in cm) away from the ball to start slowing down."""
    _CLOSE_THRESHOLD = 50

    """The multiplier for the radius to find how far away Cozmo should be from the ball."""
    _RADIUS_MULTIPLIER = 8

    # Convert to mm.
    _CLOSE_THRESHOLD *= 10

    def __init__(self, args=None):
        """
        Initializes the state when it is first switched to.

        Args:
            args: Whether Cozmo is approaching after previously hitting the ball.
        """
        self.ballTimer = self._BALL_TIMEOUT
        self.didHitBall = args

    async def update(self, owner):
        """
        Executes the state's behavior for the current tick.

        Args:
            owner: The object to affect behavior for.
        
        Returns:
            If the object's state should be changed, returns the class of the new state.
            Otherwise, return None.
        """
        robot = owner
        if owner.ball:
            radius = owner.ball[2]
            x = owner.ball[0]
            y = owner.ball[1]

            width = len(robot.opencv_image[0])

            center = width / 2

            # Adjust FOV
            direction_offset = (center - x) / center * self._ANGLE_OF_VIEW / 2

            move_by = (
                        width / 2 / radius * self._PHYSICAL_RADIUS / self._TAN_ANGLE - self._PHYSICAL_RADIUS - self._COZMO_OFFSET) * 10

            if owner.debug:
                print("Angle: " + str(direction_offset))
                print("Distance: " + str(move_by))

            robot.last_angle = direction_offset

            hit_threshold = self._PHYSICAL_RADIUS * self._RADIUS_MULTIPLIER
            if move_by < hit_threshold:
                return HitBall()

            left_speed = right_speed = self._MOVE_SPEED
            speed_difference = direction_offset / self._ANGLE_OF_VIEW * 1.5 * self._MOVE_SPEED
            if direction_offset < 0:
                right_speed += speed_difference
            elif direction_offset > 0:
                left_speed -= speed_difference

            if move_by < self._CLOSE_THRESHOLD:
                speed_multiplier = (move_by - hit_threshold) / (self._CLOSE_THRESHOLD - hit_threshold)
                speed_multiplier = max(speed_multiplier, self._MIN_SPEED_MULTIPLIER)
                left_speed *= speed_multiplier
                right_speed *= speed_multiplier

            await owner.drive_wheels(left_speed, right_speed, 500, 500)
        elif self.ballTimer >= self._BALL_TIMEOUT:
            back_up = robot.ball != None
            back_distance = -self._PHYSICAL_RADIUS * 20
            if back_up and robot.ball[2] < self._BALL_BACK_UP:
                back_distance = -back_distance
            if not back_up and self.didHitBall:
                back_up = True
            if not back_up and BallAnnotator.ball and BallAnnotator.ball[2] > self._BALL_BACK_UP:
                back_up = True

            BallAnnotator.ball = None
            if back_up:
                return BackUp(back_distance)
            else:
                return Search(robot.last_angle > 0)
        else:
            self.ballTimer += 1


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
        return PathPlan()


class BackUp(State):
    """Backs up if too close to the ball."""

    def __init__(self, args):
        """
        Initializes the state when it is first switched to.

        Args:
            args: The distance to back up.
        """
        self.back_distance = args
        pass

    async def update(self, owner):
        """
        Executes the state's behavior for the current tick.

        Args:
            owner: The object to affect behavior for.
        
        Returns:
            If the object's state should be changed, returns the class of the new state.
            Otherwise, return None.
        """
        if owner.debug:
            print("Back up")
        await doActionWithTimeout(owner.drive_straight(distance_mm(self.back_distance), speed_mmps(1000), in_parallel = True), 1)
        return Approach()


if __name__ == '__main__':
    cozmo.run_program(run, use_viewer=True, force_viewer_on_top=True)
