
#author1: Cheng Hann Gan
#author2: James Park

from grid import *
from visualizer import *
import threading
from queue import PriorityQueue
import math
import cozmo
from cozmo.util import degrees, radians, distance_mm, speed_mmps
import time
import numpy
from operator import itemgetter


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

def cozmoBehavior(robot: cozmo.robot.Robot):
    """Cozmo search behavior. See assignment document for details

        Has global access to grid, a CozGrid instance created by the main thread, and
        stopevent, a threading.Event instance used to signal when the main thread has stopped.
        You can use stopevent.is_set() to check its status or stopevent.wait() to wait for the
        main thread to finish.

        Arguments:
        robot -- cozmo.robot.Robot instance, supplied by cozmo.run_program
    """

    global grid, stopevent
    global seen_cubes

    robot.set_head_angle(degrees(0)).wait_for_completed()
    robot.set_lift_height(0).wait_for_completed()

    # The driving speed of the robot.
    ROBOT_SPEED = 100
    # The turn speed of the robot when turning.
    TURN_SPEED = 20
    # The yaw speed (degrees) of the robot when turning
    TURN_YAW = 1 / 55 * TURN_SPEED * 2
    # The acceleration of the robot.
    ROBOT_ACCELERATION = 1000

    # The cubes that can be seen by the robot on the current frame.
    seen_cubes = [None, None, None]
    # The cubes that the robot has seen.
    cubes = [None, None, None]

    # Robot position in millimeters.
    cozmo_position = [45, 27.5]
    # Robot rotation in radians.
    cozmo_rotation = 0
    # The position of the robot in grid coordinates.
    grid_position = worldToGridCoords(cozmo_position)
    # The position of the robot in grid coordinates on the previous frame.
    prev_grid_position = None
    # The world position that the robot is trying to get to.
    goal = None
    # The offset from the wall that the robot starts at.
    start_offset = [25.4, 25.4]
    cozmo_position = list(cozmo_position[i] + start_offset[i] for i in range(len(cozmo_position)))

    # The time on the previous frame.
    prev_time = time.time()
    # The time (ms) since the previous frame.
    delta_time = 0

    # Whether the robot was driving on the previous frame.
    was_driving = False
    # Whether the robot was turning with drive_wheels on the previous frame.
    was_turning = False
    # Cooldown (s) for keeping track of movement when the robot is starting to drive.
    DRIVE_COOLDOWN = 0
    # Timer (s) for keeping track of movement when the robot is starting to drive.
    drive_timer = DRIVE_COOLDOWN
    # Threshold for ignoring small turns.
    TURN_THRESHOLD = 0.5

    # Timer (s) for letting the robot look forward at the start before driving to the center.
    start_timer = 1
    
    # The next grid cell that the robot is headed for.
    next = None
    # Whether the robot or cube grid positions have changed on the current frame.
    changed = True

    # Initialize listener
    robot.world.add_event_handler(cozmo.objects.EvtObjectObserved, object_observed_handler)
    robot.world.add_event_handler(cozmo.objects.EvtObjectDisappeared, object_disappeared_handler)

    while not stopevent.is_set():
        # Update the delta time since the last frame.
        current_time = time.time()
        delta_time = current_time - prev_time
        prev_time = current_time

        cozmo_rotation %= math.pi * 2
        rotation_cos = math.cos(cozmo_rotation)
        rotation_sin = math.sin(cozmo_rotation)
        prev_grid_position = grid_position
        if was_driving:
            if start_timer > 0:
                start_timer -= delta_time
            else:
                speed_delta = delta_time * ROBOT_SPEED

                cozmo_position = (cozmo_position[0] + rotation_cos * speed_delta, cozmo_position[1] + rotation_sin * speed_delta)
                grid_position = worldToGridCoords(cozmo_position)
                grid.setStart(grid_position)
        else:
            drive_timer = DRIVE_COOLDOWN
        if was_turning:
            cozmo_rotation += TURN_YAW * delta_time

        # Check if any cubes have changed.
        for i, seen_cube in enumerate(seen_cubes):
            current_cube = cubes[i]
            seen_exists = bool(seen_cube)
            current_exists = bool(current_cube)
            current_change = False
            
            seen_object = None
            if seen_exists:
                seen_object = Cube(seen_cube, cozmo_position, cozmo_rotation)

            if seen_exists and not current_exists and grid.coordInBounds(seen_object.grid_position):
                current_change = True
            elif seen_exists and current_exists:
                if seen_object.grid_position != current_cube.grid_position or abs(seen_object.angle - current_cube.angle) > 22.5:
                    current_change = True
            if current_change:
                changed = True
                cubes[i] = seen_object

        if start_timer > 0:
            start_timer -= delta_time
            continue
        
        if not changed and prev_grid_position != grid_position:
            changed = True
        
        if cubes[0]:
            cube_position = cubes[0].position
            direction_offset = (math.cos(cubes[0].angle), math.sin(cubes[0].angle))
            goal_world = tuple(cube_position[i] - direction_offset[i] * Cube.EXPAND_SIZE * 0.75 for i in range(len(cube_position)))
            goal = worldToGridCoords(goal_world)
        else:
            goal = getGridCenter()

        if changed:
            grid.clearObstacles()
            # Cube obstacles.
            for cube in cubes:
                if cube:
                    grid_points = getGridPoints(cube.position[0], cube.position[1], cube.angle)
                    for point in grid_points:
                        if grid.coordInBounds(point):
                            grid.addObstacle(point)

            # Wall obstacles.
            for i in range(0, grid.width):
                grid.addObstacle((i, 0))
                grid.addObstacle((i, grid.height - 1))
            for i in range(1, grid.height - 1):
                grid.addObstacle((0, i))
                grid.addObstacle((grid.width - 1, i))

            grid.clearGoals()
            grid.setStart(grid_position)
            grid.addGoal(goal)
            astar(grid, heuristic)

        path = grid.getPath()
        was_turning = False
        if path != None and len(path) > 1:
            next = path[0]
            if path[0] == grid_position:
                next = path[1]

            turn = getTurnDirection(rotation_cos, rotation_sin, grid_position, next)
            if abs(turn) > TURN_THRESHOLD and abs(2 * math.pi - abs(turn)) > TURN_THRESHOLD:
                robot.stop_all_motors()
                robot.turn_in_place(radians(turn), num_retries=3).wait_for_completed()
                cozmo_rotation += turn
                was_driving = False
            else:
                robot.drive_wheels(ROBOT_SPEED, ROBOT_SPEED, ROBOT_ACCELERATION, ROBOT_ACCELERATION)
                was_driving = True
        else:
            was_driving = False
            if cubes[0]:
                turn = getTurnDirection(rotation_cos, rotation_sin, grid_position, cubes[0].grid_position)
                robot.stop_all_motors()
                if abs(turn) > TURN_THRESHOLD and abs(2 * math.pi - abs(turn)) > TURN_THRESHOLD:
                    #print(turn, grid_position, cubes[0].grid_position, cozmo_rotation)
                    robot.turn_in_place(radians(turn), num_retries=3).wait_for_completed()
                    cozmo_rotation += turn
            else:
                robot.drive_wheels(TURN_SPEED, -TURN_SPEED, ROBOT_ACCELERATION, ROBOT_ACCELERATION)
                was_turning = True
        changed = False

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

def getCubeID(cube):
    """
    Maps a cube's object_id field to its cube number.
   
    Args:
        cube: The cube to get an ID for.
       
    Returns:
        The ID of the cube.
    """
    object_id = cube.object_id
    light_cubes = cube.world.light_cubes
 
    if light_cubes[cozmo.objects.LightCube1Id].object_id == object_id:
        return 0
    if light_cubes[cozmo.objects.LightCube2Id].object_id == object_id:
        return 1
    if light_cubes[cozmo.objects.LightCube3Id].object_id == object_id:
        return 2
   
    return -1

class Cube:

    """The actual width (mm) of the cube."""
    ACTUAL_SIZE = 45
    """The expanded width (mm) of the cube, taking the robot's size into account to make the robot a point."""
    EXPAND_SIZE = ACTUAL_SIZE + (90 ** 2 + 55 ** 2) ** 0.5

    def __init__(self, cube, cozmo_position = (0, 0), cozmo_rotation = 0):
        """
        Initializes the cube.

        Args:
            cube: The cube observation to get pose data from.
            cozmo_position: The position of Cozmo.
            cozmo_rotation: The rotation of Cozmo.
        """
        self.angle = cube.pose.rotation.angle_z.radians
        posX = cube.pose.position.x + 45 + Cube.ACTUAL_SIZE / 2
        posY = cube.pose.position.y + 27.5 + Cube.ACTUAL_SIZE / 2
        self.position = (posX, posY)
        self.grid_position = worldToGridCoords(self.position)

def getGridPoints(x, y, angle):
    # Set up
    grid_size = 25
    x /= grid_size
    y /= grid_size
    SQUARE_WIDTH = Cube.EXPAND_SIZE / grid_size
    half_width = SQUARE_WIDTH / 2.0
    cos_offset = half_width * math.cos(angle)
    sin_offset = half_width * math.sin(angle)

    # Calculated corners of a rectangle after rotation
    corners = [(x + cos_offset - sin_offset, y + cos_offset + sin_offset),
               (x - cos_offset - sin_offset, y + cos_offset - sin_offset),
               (x - cos_offset + sin_offset, y - cos_offset - sin_offset),
               (x + cos_offset + sin_offset, y - cos_offset + sin_offset)]

    # Generate lines from given corners
    lines = []
    last = None
    for p in corners:
        if last is not None:
            lines.append((last, p))
        last = p
    lines.append((corners[len(corners) - 1], corners[0]))

    # Get min & max scan range
    min_point = (min(corners, key=itemgetter(0))[0], min(corners, key=itemgetter(1))[1])
    max_point = (max(corners, key=itemgetter(0))[0], max(corners, key=itemgetter(1))[1])

    # For all possible points that may be in the polygon (or square),
    # append to points
    points = []
    for y in range(math.floor(min_point[1]), math.ceil(max_point[1])):
        for x in range(math.floor(min_point[0]), math.ceil(max_point[0])):
            if pointInsidePolygonLines((x, y), lines):
                points.append((x, y))

    return points
    # end. : All intersecting grid coordinates are recorded to points

def object_observed_handler(evt, image_box=None, obj=None, pose=None, updated=None, **kwargs):
    id = getCubeID(obj)
    seen_cubes[id] = obj


def object_disappeared_handler(evt, obj=None, **kwargs):
    id = getCubeID(obj)
    seen_cubes[id] = None

def worldToGridCoords(worldCoord):
    """
    Converts world coordinates to grid coordinates.

    Args:
        worldCoord: The world coordinates to convert to grid coordinates.

    Returns:
        The grid coordinates corresponding to the world coordinates
        The grid's center if the coordinate is out of bounds.
    """
    gridCoord = tuple(int(worldCoord[i] / grid.scale) for i in range(len(worldCoord)))
    if grid.coordInBounds(gridCoord):
        return gridCoord
    return getGridCenter()

def getGridCenter():
    """
    Gets the coordinates of the center of the grid.

    Returns:
        The coordinates of the center of the grid.
    """
    return (int(grid.width / 2), int(grid.height / 2))

"""
 * Licensed under the Apache License, Version 2.0 (the "License");
 *     http://www.apache.org/licenses/LICENSE-2.0
"""
def rayTrace(p1, p2, line):
    p1, p2, p3, p4 = line[0], line[1], p1, p2
    p = getIntersectPoint(p1, p2, p3, p4)
    if p is not None:
        p = p[0]
        if between(p[0], p1[0], p2[0]) and between(p[1], p1[1], p2[1]) and between(p[0], p3[0], p4[0]) and between(p[1], p3[1], p4[1]):
            return p
    return None


# Calc the point 'b' where line crosses the Y axis
def calculateYAxisIntersect(p, m):
    return p[1] - (m * p[0])


# Checks if the first number is between the other two numbers.
# Also returns true if all numbers are very close together to the point where they are essentially equal
# (i.e., floating point approximation).
def between(p, p1, p2):
    return p + 0.00000001 >= min(p1, p2) and p - 0.00000001 <= max(p1, p2)

# Calc the gradient 'm' of a line between p1 and p2
def calculateGradient(p1, p2):
    # Ensure that the line is not vertical
    if p1[0] != p2[0]:
        m = (p1[1] - p2[1]) / float(p1[0] - p2[0])
        return m
    return None


def getIntersectPoint(p1, p2, p3, p4):
    m1 = calculateGradient(p1, p2)
    m2 = calculateGradient(p3, p4)

    # See if the the lines are parallel
    if m1 != m2:
        # Not parallel

        # See if either line is vertical
        if m1 is not None and m2 is not None:
            # Neither line vertical
            b1 = calculateYAxisIntersect(p1, m1)
            b2 = calculateYAxisIntersect(p3, m2)
            x = (b2 - b1) / float(m1 - m2)
            y = (m1 * x) + b1
        else:
            # Line 1 is vertical so use line 2's values
            if m1 is None:
                b2 = calculateYAxisIntersect(p3, m2)
                x = p1[0]
                y = (m2 * x) + b2
            # Line 2 is vertical so use line 1's values
            elif m2 is None:
                b1 = calculateYAxisIntersect(p1, m1)
                x = p3[0]
                y = (m1 * x) + b1
            else:
                assert false

        return (x, y),
    else:
        # Parallel lines with same 'b' value must be the same line so they intersect
        # everywhere in this case we return the start and end points of both lines
        # the calculateIntersectPoint method will sort out which of these points
        # lays on both line segments
        b1, b2 = None, None  # vertical lines have no b value
        if m1 is not None:
            b1 = calculateYAxisIntersect(p1, m1)

        if m2 is not None:
            b2 = calculateYAxisIntersect(p3, m2)

        # If these parallel lines lay on one another
        if b1 == b2:
            return p1, p2, p3, p4
        return None


# Determine whether a point is inside an simple polygon. Polygon is a set of lines.
def pointInsidePolygonLines(point, polygon):
    count = 0
    for l in polygon:
        result = rayTrace(point, (-10, -100 / 2.0), l)
        if result is not None:
            if between(result[0], point[0], point[0]) and between(result[1], point[1], point[1]):
                return True
            count += 1
    return count % 2 == 1

######################## DO NOT MODIFY CODE BELOW THIS LINE ####################################


class RobotThread(threading.Thread):
    """Thread to run cozmo code separate from main thread
    """
        
    def __init__(self):
        threading.Thread.__init__(self, daemon=True)

    def run(self):
        cozmo.run_program(cozmoBehavior)


# If run as executable, start RobotThread and launch visualizer with empty grid file
if __name__ == "__main__":
    global grid, stopevent
    stopevent = threading.Event()
    grid = CozGrid("emptygrid.json")
    visualizer = Visualizer(grid)
    updater = UpdateThread(visualizer)
    updater.start()
    robot = RobotThread()
    robot.start()
    visualizer.start()
    stopevent.set()