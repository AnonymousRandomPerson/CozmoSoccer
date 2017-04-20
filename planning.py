
#author1: Cheng Hann Gan
#author2: James Park

from grid import *
import threading
from queue import PriorityQueue
import math
import cozmo
from cozmo.util import degrees, radians, distance_mm, speed_mmps
import time
import numpy
from operator import itemgetter

from state_machine import State

from utils import *

import goto_ball

def astar(grid, heuristic):
    """Perform the A* search algorithm on a defined grid

        Arguments:
        grid -- CozGrid instance to perform search on
        heuristic -- supplied heuristic function
    """
    grid.clearVisited()
    grid.clearPath()
    last = None
    queue = PriorityQueue()

    init = grid.getStart()
    goals = grid.getGoals()
    if len(goals) == 0:
        return
    goal = goals[0]
    if not goal or not init:
        return

    # (totalDistance, node, previous, startDistance)
    queue.put((0, init, None, heuristic(init, goal)))
    done = False
    while not done:
        if queue.empty():
            done = True
        else:
            current = queue.get()
            if current[1] == goal:
                last = current
                done = True
            else:
                visited = grid.getVisited()
                if not current[1] in visited:
                    currentSuccessors = grid.getNeighbors(current[1])
                    for successor in currentSuccessors:
                        coordinate = successor[0]
                        if not coordinate in visited:
                            startDistance = current[3] + successor[1]
                            queue.put((startDistance + heuristic(coordinate, goal), coordinate, current, startDistance))
                    grid.addVisited(current[1])

    # Reconstruct the path.
    if last:
        current = last
        path = []
        while current:
            path += [current[1]]
            current = current[2]
        path = list(reversed(path))
        grid.setPath(path)


def heuristic(current, goal):
    """Heuristic function for A* algorithm

        Arguments:
        current -- current cell
        goal -- desired goal cell
    """
    distance = getDistance(current, goal)
    return distance

def getDistance(pos1, pos2):
    """
    Gets the distance between two points.

    Args:
        pos1: The first point to get the distance between.
        pos2: The second point to get the distance between.
    
    Returns:
        The distance between the two points.
    """
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

class PathPlan(State):
    """Plans a path to the side of the ball away from the goal."""

    async def update(self, robot):
        """
        Executes the state's behavior for the current tick.

        Args:
            robot: The object to affect behavior for.
        
        Returns:
            If the object's state should be changed, returns the class of the new state.
            Otherwise, return None.
        """

        robot.rotation %= math.pi * 2
        rotation_cos = math.cos(robot.rotation)
        rotation_sin = math.sin(robot.rotation)
        if robot.was_driving:
            speed_delta = robot.delta_time * robot.ROBOT_SPEED

            robot.position = (robot.position[0] + rotation_cos * robot.speed_delta, robot.position[1] + rotation_sin * robot.speed_delta)
            robot.grid_position = robot.grid.worldToGridCoords(robot.position)
            robot.grid.setStart(robot.grid_position)
        else:
            robot.drive_timer = robot.DRIVE_COOLDOWN
        if robot.was_turning:
            robot.rotation += robot.TURN_YAW * robot.delta_time

        changed = False
        if robot.ball:
            if robot.prev_ball:
                robot.ball_grid = robot.grid.worldToGridCoords(robot.ball)
                robot.ball_prev_grid = robot.grid.worldToGridCoords(robot.prev_ball)
                changed = robot.ball_grid != robot.ball_prev_grid
            else:
                changed = True
        
        if not changed and robot.prev_grid_position != robot.grid_position:
            changed = True

        if changed:
            robot.grid.clearObstacles()
            if robot.ball:
                grid_points = getGridPoints(robot.ball_grid[0], robot.ball_grid[1], robot)
                for point in grid_points:
                    if robot.grid.coordInBounds(point):
                        print(point)
                        robot.grid.addObstacle(point)

            # Wall obstacles.
            for i in range(0, robot.grid.width):
                robot.grid.addObstacle((i, 0))
                robot.grid.addObstacle((i, robot.grid.height - 1))
            for i in range(1, robot.grid.height - 1):
                robot.grid.addObstacle((0, i))
                robot.grid.addObstacle((robot.grid.width - 1, i))

            robot.grid.clearGoals()
            robot.grid.setStart(robot.grid_position)
            robot.grid.addGoal(robot.target_position)
            astar(robot.grid, heuristic)

        path = robot.grid.getPath()
        robot.was_turning = False
        if path != None and len(path) > 1:
            robot.next_cell = path[0]
            if path[0] == robot.grid_position:
                robot.next_cell = path[1]

            turn = getTurnDirection(rotation_cos, rotation_sin, robot.grid_position, robot.next_cell)
            if abs(turn) > robot.TURN_THRESHOLD and abs(2 * math.pi - abs(turn)) > robot.TURN_THRESHOLD:
                robot.stop_all_motors()
                await robot.turn_in_place(radians(turn), num_retries=3).wait_for_completed()
                robot.rotation += turn
                robot.was_driving = False
            else:
                await robot.drive_wheels(robot.ROBOT_SPEED, robot.ROBOT_SPEED, robot.ROBOT_ACCELERATION, robot.ROBOT_ACCELERATION)
                robot.was_driving = True
        else:
            robot.was_driving = False
            if robot.ball:
                turn = getTurnDirection(rotation_cos, rotation_sin, robot.grid_position, robot.ball_grid)
                robot.stop_all_motors()
                if abs(turn) > robot.TURN_THRESHOLD and abs(2 * math.pi - abs(turn)) > robot.TURN_THRESHOLD:
                    await robot.turn_in_place(radians(turn), num_retries=3).wait_for_completed()
                    robot.rotation += turn
                    return goto_ball.Approach()
            else:
                await robot.drive_wheels(robot.TURN_SPEED, -robot.TURN_SPEED, robot.ROBOT_ACCELERATION, robot.ROBOT_ACCELERATION)
                robot.was_turning = True

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
    forward = (round(rotation_cos), round(rotation_sin))
    target_direction = tuple(next[i] - current[i] for i in range(len(next)))
    turn = math.atan2(target_direction[1], target_direction[0]) - math.atan2(forward[1], forward[0])
    return turn

def getGridPoints(x, y, robot):
    """
    Gets the points on the grid that are blocked by the ball.

    Args:
        x: The x grid coordinate of the ball.
        y: The y grid coordinate of the ball.
        robot: The robot.

    Returns:
        The points on the grid that are blocked by the ball.
    """
    roundedGrid = (int(x), int(y))
    total_radius = (robot.RADIUS + robot.BALL_RADIUS) / robot.grid.scale
    scanAmount = math.ceil(total_radius)
    scan = range(-scanAmount, scanAmount + 1)
    corners = ((0, 0), (0, 1), (1, 1), (1, 0))
    points = []
    for i in scan:
        for j in scan:
            for corner in corners:
                if grid_distance(roundedGrid[0] + i + corner[0], roundedGrid[1] + j + corner[1], x, y) > total_radius:
                    points.append((i, j))

    return points