#!/usr/bin/env python3

''' Get a raw frame from camera and display in OpenCV
By press space, save the image from 001.bmp to ...
'''

import cv2
import cozmo
import numpy as np
from numpy.linalg import inv
import threading
import time

from ar_markers.hamming.detect import detect_markers

from grid import CozGrid
from gui import GUIWindow
from particle import Particle, Robot
from setting import *
from particle_filter import *
from utils import *
from cozmo.util import degrees, radians

from state_machine import State

# camera params
camK = np.matrix([[295, 0, 160], [0, 295, 120], [0, 0, 1]], dtype='float32')

#marker size in inches
marker_size = 3.5

# tmp cache
last_pose = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))
flag_odom_init = False

# goal location for the robot to drive to, (x, y, theta)
goal = (6,10,0)

# map
global grid, gui

async def image_processing(robot):

    global camK, marker_size

    event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

    # convert camera image to opencv format
    opencv_image = np.asarray(event.image)
    
    # detect markers
    markers = detect_markers(opencv_image, marker_size, camK)
    
    # show markers
    for marker in markers:
        marker.highlite_marker(opencv_image, draw_frame=True, camK=camK)
        #print("ID =", marker.id);
        #print(marker.contours);
    #cv2.imshow("Markers", opencv_image)

    return markers

#calculate marker pose
def cvt_2Dmarker_measurements(ar_markers):
    
    marker2d_list = [];
    
    for m in ar_markers:
        R_1_2, J = cv2.Rodrigues(m.rvec)
        R_1_1p = np.matrix([[0,0,1], [0,-1,0], [1,0,0]])
        R_2_2p = np.matrix([[0,-1,0], [0,0,-1], [1,0,0]])
        R_2p_1p = np.matmul(np.matmul(inv(R_2_2p), inv(R_1_2)), R_1_1p)
        #print('\n', R_2p_1p)
        yaw = -math.atan2(R_2p_1p[2,0], R_2p_1p[0,0])
        
        x, y = m.tvec[2][0] + 0.5, -m.tvec[0][0]
        #print('x =', x, 'y =', y,'theta =', yaw)
        
        # remove any duplate markers
        dup_thresh = 2.0
        find_dup = False
        for m2d in marker2d_list:
            if grid_distance(m2d[0], m2d[1], x, y) < dup_thresh:
                find_dup = True
                break
        if not find_dup:
            marker2d_list.append((x,y,math.degrees(yaw)))

    return marker2d_list


#compute robot odometry based on past and current pose
def compute_odometry(curr_pose, cvt_inch=True):
    global last_pose, flag_odom_init
    last_x, last_y, last_h = last_pose.position.x, last_pose.position.y, \
        last_pose.rotation.angle_z.degrees
    curr_x, curr_y, curr_h = curr_pose.position.x, curr_pose.position.y, \
        curr_pose.rotation.angle_z.degrees
    
    dx, dy = rotate_point(curr_x-last_x, curr_y-last_y, -last_h)
    if cvt_inch:
        dx, dy = dx / 25.6, dy / 25.6

    return (dx, dy, diff_heading_deg(curr_h, last_h))

#particle filter functionality
class ParticleFilter:

    def __init__(self, grid):
        self.particles = Particle.create_random(PARTICLE_COUNT, grid)
        self.grid = grid

    def update(self, odom, r_marker_list):

        # ---------- Motion model update ----------
        self.particles = motion_update(self.particles, odom)

        # ---------- Sensor (markers) model update ----------
        self.particles = measurement_update(self.particles, r_marker_list, self.grid)

        # ---------- Show current state ----------
        # Try to find current best estimate for display
        m_x, m_y, m_h, m_confident = compute_mean_pose(self.particles)
        return (m_x, m_y, m_h, m_confident)

def worldToGridCoords(worldCoord):
    """
    Converts world coordinates to grid coordinates.

    Args:
        worldCoord: The world coordinates to convert to grid coordinates.

    Returns:
        The grid coordinates corresponding to the world coordinates
        The grid's center if the coordinate is out of bounds.
    """
    gridCoord = tuple(worldCoord[i] / grid.scale for i in range(len(worldCoord)))
    return gridCoord

async def run(robot: cozmo.robot.Robot):

    global flag_odom_init, last_pose

    # Obtain odometry information
    last_pose = robot.last_pose
    current_odom = compute_odometry(robot.pose)
    robot.last_pose = robot.pose

    # Obtain list of currently seen markers and their poses
    images = await image_processing(robot)
    r_marker_list = cvt_2Dmarker_measurements(images)
    # Update the particle filter using the above information
    result = robot.pf.update(current_odom, r_marker_list)

    # Determine the robot's action based on current state of localization system
    # Then do the below.
    converged = result[3]
    if robot.played_goal_animation:
        robot.stop_all_motors()
    elif converged:
        # Have the robot drive to the goal.
        # Goal is defined w/ position & orientation
        position = result[0:2]
        rounded_position = tuple(int(position[i]) for i in range(len(position)))
        angle = math.radians(result[2])
        if robot.found_goal:
            target_angle = 0
        else:
            target_angle = math.degrees(getTurnDirection(math.cos(angle), math.sin(angle), position, goal[0:2]))
        at_goal = True

        goal_position = tuple(goal[0:2])
        for i in range(len(rounded_position)):
            if abs(rounded_position[i] - goal_position[i]) > 1:
                at_goal = False
                break
        if at_goal:
            robot.stop_all_motors()
            await robot.turn_in_place(degrees(-result[2]), num_retries=3).wait_for_completed()
            robot.found_goal = True
            if not robot.played_goal_animation:
                # Then robot should play a happy animation, and stand still.
                await robot.say_text("Yay", play_excited_animation=True, duration_scalar=0.5, voice_pitch = 1).wait_for_completed()
                robot.played_goal_animation = True
        elif abs(target_angle) > robot.TURN_TOLERANCE and abs(2 * math.pi - abs(target_angle)) > robot.TURN_TOLERANCE:
            robot.stop_all_motors()
            await robot.turn_in_place(degrees(target_angle), num_retries=3).wait_for_completed()
        else:
            robot.found_goal = False
            await robot.drive_wheels(robot.ROBOT_SPEED, robot.ROBOT_SPEED, robot.ROBOT_ACCELERATION, robot.ROBOT_ACCELERATION)
    else:
        # Have robot actively look around if localization has not converged.
        await robot.drive_wheels(robot.TURN_SPEED, -robot.TURN_SPEED, robot.ROBOT_ACCELERATION, robot.ROBOT_ACCELERATION)

    # Make code robust to "kidnapped robot problem"
    # Reset localization if robot is picked up.
    # Make robot unhappy.
    if robot.is_picked_up:
        robot.pf.__init__(robot.grid)
        robot.found_goal = False
        robot.played_goal_animation = False
        if not robot.played_angry_animation:
            await play_angry(robot)
        robot.played_angry_animation = True
        await robot.set_head_angle(cozmo.util.degrees(robot.HEAD_ANGLE)).wait_for_completed()
    else:
        robot.played_angry_animation = False

    # Update the particle filter GUI for debugging
    robot.gui.show_particles(robot.pf.particles)
    robot.gui.show_mean(*result)
    robot.gui.updated.set()

def getTurnDirection(rotation_cos, rotation_sin, current, next):
    """
    Gets the direction that the robot needs to turn to face a position.

    Args:
        rotation_cos: The cosine of the robot's rotation.
        rotation_sin: The sine of the robot's rotation.
        current: The current position of the robot.
        next: The target of the robot.

    Returns:
        The direction that the robot needs to turn to face a position.
    """
    forward = (rotation_cos, rotation_sin)
    target_direction = np.subtract(next, current)
    turn = math.atan2(target_direction[1], target_direction[0]) - math.atan2(forward[1], forward[0])
    return turn

async def play_angry(robot):
    """
    Plays an angry animation when the robot is picked up.

    Args:
        robot: The robot to play an angry animation with.
    """
    await robot.say_text("PUT ME DOWN", duration_scalar=1.5, voice_pitch=1, num_retries=2).wait_for_completed()
    await robot.set_lift_height(1, 10000).wait_for_completed()
    await robot.set_lift_height(-1, 10000).wait_for_completed()

class FindLocation(State):
    """Localizes the robot with a particle filter."""

    async def update(self, owner):
        """
        Executes the state's behavior for the current tick.

        Args:
            owner: The object to affect behavior for.
        
        Returns:
            If the object's state should be changed, returns the class of the new state.
            Otherwise, return None.
        """
        await run(owner)
        return None