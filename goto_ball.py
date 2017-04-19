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

try:
    from PIL import ImageDraw, ImageFont
except ImportError:
    sys.exit('run `pip3 install --user Pillow numpy` to run this example.')

from state_machine import State, StateMachine


# Define a decorator as a subclass of Annotator; displays battery voltage
class BatteryAnnotator(cozmo.annotate.Annotator):
    def apply(self, image, scale):
        d = ImageDraw.Draw(image)
        bounds = (0, 0, image.width, image.height)
        batt = self.world.robot.battery_voltage
        if not self.world.robot.stateMachine: return

        text = cozmo.annotate.ImageText(self.world.robot.stateMachine.getState(), color='green')
        text.render(d, bounds)


# Define a decorator as a subclass of Annotator; displays the ball
class BallAnnotator(cozmo.annotate.Annotator):
    ball = None

    def apply(self, image, scale):
        d = ImageDraw.Draw(image)
        bounds = (0, 0, image.width, image.height)

        if BallAnnotator.ball is None: return

        # double size of bounding box to match size of rendered image
        scaledBall = np.multiply(BallAnnotator.ball, 2)

        # define and display bounding box with params:
        # msg.img_topLeft_x, msg.img_topLeft_y, msg.img_width, msg.img_height
        box = cozmo.util.ImageBox(scaledBall[0] - scaledBall[2],
                                  scaledBall[1] - scaledBall[2],
                                  scaledBall[2] * 2, scaledBall[2] * 2)
        cozmo.annotate.add_img_box_to_image(image, box, "green", text=None)


async def run(robot: cozmo.robot.Robot):
    '''The run method runs once the Cozmo SDK is connected.'''

    # add annotators for battery level and ball bounding box
    stateMachine = StateMachine(robot)
    await stateMachine.changeState(Search())
    robot.stateMachine = stateMachine
    robot.world.image_annotator.add_annotator('battery', BatteryAnnotator)
    robot.world.image_annotator.add_annotator('ball', BallAnnotator)
    robot.debug = False

    # Camera settings
    robot.camera.color_image_enabled = True
    robot.camera.set_manual_exposure(10, 3.9)

    await robot.set_head_angle(degrees(-5), in_parallel = True).wait_for_completed()
    await robot.set_lift_height(1.0, in_parallel = True).wait_for_completed()

    # The number of images that the robot will take into account to get the ball's average data.
    NUM_TRIALS = 2
    # Whether to disable the robot from moving.
    MOTION_DISABLED = False

    robot.last_angle = 0

    try:
        while True:
            # get camera image
            event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

            # convert camera image to opencv format
            robot.opencv_image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_RGB2BGR)

            # find the ball
            ball = find_ball.find_ball(cv2.cvtColor(np.asarray(event.image), cv2.COLOR_RGB2GRAY))
            # @TODO Testing
            goal = find_goal.find_goal(robot, robot.opencv_image)

            # set annotator ball
            if robot.debug:
                print("Ball: " + str(ball))
            BallAnnotator.ball = ball
            if ball and ball[1] > len(robot.opencv_image[0]) / 10:
                robot.ball = ball
            else:
                robot.ball = None

            if not MOTION_DISABLED:
                await stateMachine.update()

    except KeyboardInterrupt:
        print("")
        print("Exit requested by user")
    except cozmo.RobotBusy as e:
        print(e)


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
    await robot.set_lift_height(0, 100, 100, in_parallel = True).wait_for_completed()
    await robot.set_lift_height(1, 100, 100, in_parallel = True).wait_for_completed()


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

    """The amount to turn each time the robot turns while searching for a ball."""
    _TURN_SPEED = 0
    """The direction that the robot will turn in."""
    _TURN_DIRECTION = 1

    def __init__(self, args=None):
        """Initializes the state when it is first switched to.

        Args:
            args: Whether the robot will spin counterclockwise.
        """
        if args:
            self._TURN_DIRECTION = -1

    async def update(self, owner):
        """Executes the state's behavior for the current tick.

        Args:
            owner: The object to affect behavior for.
        
        Returns:
            If the object's state should be changed, returns the class of the new state.
            Otherwise, return None.
        """
        if owner.ball:
            await owner.drive_wheels(0, 0, 500, 500)
            return Approach()
        turn_speed = self._TURN_SPEED * self._TURN_DIRECTION
        await owner.drive_wheels(turn_speed, -turn_speed, 500, 500)


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
        await owner.drive_wheels(0, 0, 500, 500)
        await hitBall(owner)
        return Approach(True)


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
