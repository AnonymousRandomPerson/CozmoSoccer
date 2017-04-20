import cozmo
import threading
import time

import go_to_goal
import goto_ball
import planning
from grid import CozGrid
from gui import GUIWindow

from state_machine import State, StateMachine

global grid, gui
Map_filename = "map_arena.json"
grid = CozGrid(Map_filename)
show_gui = False
if show_gui:
    gui = GUIWindow(grid)
else:
    gui = None

async def run(robot: cozmo.robot.Robot):
    """
    Causes the robot to play one-robot soccer.

    Args:
        robot: The robot to play soccer with.
    """
    await initialize_robot(robot)

    # start streaming
    robot.camera.image_stream_enabled = True

    robot.stateMachine = StateMachine(robot)
    await robot.stateMachine.changeState(goto_ball.Search())

    await robot.set_head_angle(cozmo.util.degrees(robot.HEAD_ANGLE)).wait_for_completed()
    await robot.set_lift_height(0, 10000).wait_for_completed()

    # Start particle filter.
    robot.pf = go_to_goal.ParticleFilter(grid)

    while True:
        # Update the delta time since the last frame.
        current_time = time.time()
        robot.delta_time = current_time - robot.prev_time
        robot.prev_time = current_time

        robot.grid_position = robot.grid.worldToGridCoords(robot.position)
        robot.prev_grid_position = robot.grid.worldToGridCoords(robot.prev_position)
        if robot.gui:
            robot.gui.show_mean(robot.grid_position[0], robot.grid_position[1], robot.rotation)

        if robot.ball:
            robot.ball_grid = robot.grid.worldToGridCoords(robot.ball)
        else:
            robot.ball_grid = None
        if robot.prev_ball:
            robot.prev_ball_grid = robot.grid.worldToGridCoords(robot.prev_ball)
        else:
            robot.prev_ball_grid = None

        await robot.stateMachine.update()

        robot.prev_ball = robot.ball
        robot.prev_position = robot.position
        robot.last_pose = robot.pose

        if robot.gui:
            robot.gui.updated.set()

async def initialize_robot(robot):
    """
    Initializes fields for the robot.

    Args:
        robot: The robot to initialize fields for.
    """
    global grid, gui
    robot.gui = gui
    robot.grid = grid

    # The angle that the robot's head faces.
    robot.HEAD_ANGLE = 5

    # The driving speed of the robot.
    robot.ROBOT_SPEED = 40
    # The turn speed of the robot when turning.
    robot.TURN_SPEED = 20
    # The yaw speed (degrees) of the robot when turning
    robot.TURN_YAW = 1 / 55 * robot.TURN_SPEED * 2
    # The acceleration of the robot.
    robot.ROBOT_ACCELERATION = 1000
    # The amount of difference between the target and actual angles that the robot will tolerate when turning.
    robot.TURN_TOLERANCE = 20

    # The length of the robot in mm.
    robot.LENGTH = 90
    # The width of the robot in mm.
    robot.WIDTH = 55
    # The maximum diagonal radius of the robot in mm.
    robot.RADIUS = (robot.LENGTH ** 2 + robot.WIDTH ** 2) ** 0.5
    # The radius of the ball in mm.
    robot.BALL_RADIUS = 20

    # The horizontal angle of view of the robot's camera (in degrees).
    robot.ANGLE_OF_VIEW = 60

    # Robot position in millimeters.
    robot.position = [45, 27.5]
    # Robot rotation in radians.
    robot.rotation = 0
    # The position of the robot in the previous frame.
    robot.prev_position = robot.position
    # The world position that the robot is trying to get to.
    robot.target_position = None
    # The world position of the goal.
    robot.goal = None

    # The world position of the ball.
    robot.ball = None
    # The previous world position of the ball.
    robot.prev_ball = None

    # The time on the previous frame.
    robot.prev_time = time.time()
    # The time (ms) since the previous frame.
    robot.delta_time = 0

    # Whether the robot has reached the goal.
    robot.found_goal = False
    # Whether the robot has played an animation upon reaching the goal.
    robot.played_goal_animation = False
    # Whether the robot has played an animation upon being picked upon.
    robot.played_angry_animation = False

    # Whether the robot was driving on the previous frame.
    robot.was_driving = False
    # Whether the robot was turning with drive_wheels on the previous frame.
    robot.was_turning = False
    # Cooldown (s) for keeping track of movement when the robot is starting to drive.
    robot.DRIVE_COOLDOWN = 0
    # Timer (s) for keeping track of movement when the robot is starting to drive.
    robot.drive_timer = robot.DRIVE_COOLDOWN
    # Threshold for ignoring small turns.
    robot.TURN_THRESHOLD = 0.5

    # The next grid cell that the robot is headed for.
    robot.next_cell = None

    # The robot's pose in the last frame.
    robot.last_pose = robot.pose

    # The current grid position of the ball.
    robot.ball_grid = (0, 0)
    # The previous grid position of the ball.
    robot.prev_ball_grid = (0, 0)
    # The current grid position of the robot.
    robot.grid_position = (0, 0)
    # The previous grid position of the robot.
    robot.prev_grid_position = (0, 0)

class CozmoThread(threading.Thread):
    """Thread for robot action execution."""
    
    def __init__(self):
        """Initializes the thread."""
        threading.Thread.__init__(self, daemon=False)

    def run(self):
        """Executes the thread."""
        cozmo.robot.Robot.drive_off_charger_on_connect = False  # Cozmo can stay on his charger
        cozmo.run_program(run, use_viewer=False)

if __name__ == '__main__':

    # cozmo thread
    cozmo_thread = CozmoThread()
    cozmo_thread.start()

    # init
    if gui:
        gui.start()