import cozmo
import threading

import go_to_goal
from grid import CozGrid
from gui import GUIWindow

from state_machine import State, StateMachine

global grid, gui
Map_filename = "map_arena.json"
grid = CozGrid(Map_filename)
gui = GUIWindow(grid)

async def run(robot: cozmo.robot.Robot):
    """
    Causes the robot to play one-robot soccer.

    Args:
        robot: The robot to play soccer with.
    """
    global grid, gui
    # start streaming
    robot.camera.image_stream_enabled = True
    robot.gui = gui
    robot.grid = grid

    stateMachine = StateMachine(robot)
    await stateMachine.changeState(go_to_goal.FindLocation())

    robot.HEAD_ANGLE = 5

    await robot.set_head_angle(cozmo.util.degrees(robot.HEAD_ANGLE)).wait_for_completed()
    await robot.set_lift_height(-1, 10000).wait_for_completed()

    #start particle filter
    robot.pf = go_to_goal.ParticleFilter(grid)

    # The driving speed of the robot.
    robot.ROBOT_SPEED = 40
    # The turn speed of the robot when turning.
    robot.TURN_SPEED = 20
    # The acceleration of the robot.
    robot.ROBOT_ACCELERATION = 1000
    # The amount of difference between the target and actual angles that the robot will tolerate when turning.
    robot.TURN_TOLERANCE = 20

    robot.found_goal = False
    robot.played_goal_animation = False
    robot.played_angry_animation = False
    
    robot.last_pose = robot.pose

    while True:
        await stateMachine.update()
        #await go_to_goal.run(robot)

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
    gui.start()