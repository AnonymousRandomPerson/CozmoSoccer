#!/usr/bin/env python3

import cv2
import sys
import copy

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit('install Pillow to run this code')


def find_goal(robot, opencv_image, debug=True):
    """Find the goal in an image.

        Arguments:
        opencv_image -- the image
        debug -- an optional argument which can be used to control whether
                debugging information is displayed.

        Returns [x, y, area] of the goal, and None if no goal is found.
    """
    cv2.waitKey(1)
    # Process Image
    robot.camera.set_manual_exposure(10, 3.9)
    opencv_image = cv2.bilateralFilter(opencv_image, 7, 50, 50)
    mask = cv2.inRange(opencv_image, np.array(
        [25, 25, 150]), np.array([100, 100, 255]))
    #opencv_image = cv2.bitwise_and(opencv_image, opencv_image, mask = mask)

    canny_image = cv2.Canny(mask, 0, 50, apertureSize=3)
    cv2.imshow('canny', canny_image)
    """
    lines = cv2.HoughLinesP(canny_image, 1, np.pi/180, 100, minLineLength=15, maxLineGap=25)
    print(lines)
    if lines is None:
        return None
    for x_1, y_1, x_2, y_2 in lines[0]:
        cv2.line(opencv_image, (x_1, y_1), (x_2, y_2), (0, 255, 0), 2)
    cv2.imshow('lines', opencv_image)
    """

    b, contours, hierarchy = cv2.findContours(
        canny_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Keep top 10
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        # Approximation precision @ 1%
        cnt = cv2.approxPolyDP(cnt, 0.01 * cnt_len, True)
        cnt_area = cv2.contourArea(cnt)
        cnt_len = len(cnt)
        # @todo contour area check
        # and cv2.isContourConvex(cnt):
        if cnt_len > 3 and cnt_len < 13 and cnt_area > 200:

            # Calculate average of y of all points
            sum_y = 0.0
            for point in cnt:
                sum_y += point[0][1]
            avg_y = sum_y / cnt_len
            min_point, max_point = cnt[0], cnt[0]
            for point in cnt:
                if point[0][1] > avg_y:
                    continue
                min_point = min(min_point, point, key=lambda p: p[0][0])
                max_point = max(max_point, point, key=lambda p: p[0][0])

            if find_angle(min_point, max_point) > 20:
                continue

            x_1, y_1 = min_point[0][0], min_point[0][1]
            x_2, y_2 = max_point[0][0], max_point[0][1]

            cv2.line(opencv_image, (x_1, y_1), (x_2, y_2), (0, 255, 0), 2)
            #cv2.drawContours(opencv_image, [cnt], -1, (255, 0, 0), 1)
            cv2.imshow('processed img', opencv_image)
            # return [*center, 50]

    return None


def find_angle(p_1, p_2):
    x_1, y_1 = p_1[0][0], p_1[0][1]
    x_2, y_2 = p_2[0][0], p_2[0][1]
    return abs(np.rad2deg(np.arctan2(y_2 - y_1, x_2 - x_1)))


def display_circles(opencv_image, circles, best=None):
    """Display a copy of the image with superimposed circles.

       Provided for debugging purposes, feel free to edit as needed.

       Arguments:
        opencv_image -- the image
        circles -- list of circles, each specified as [x,y,radius]
        best -- an optional argument which may specify a single circle that will
                be drawn in a different color.  Meant to be used to help show which
                circle is ranked as best if there are multiple candidates.

    """
    # make a copy of the image to draw on
    circle_image = copy.deepcopy(opencv_image)
    circle_image = cv2.cvtColor(circle_image, cv2.COLOR_GRAY2RGB, circle_image)

    for c in circles:
        # draw the outer circle
        cv2.circle(circle_image, (c[0], c[1]), c[2], (255, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(circle_image, (c[0], c[1]), 2, (0, 255, 255), 3)
        # write coords
        cv2.putText(circle_image, str(c), (c[0], c[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255, 255, 255), 2, cv2.LINE_AA)

    # highlight the best circle in a different color
    if best is not None:
        # draw the outer circle
        cv2.circle(circle_image, (best[0], best[1]), best[2], (0, 0, 255), 2)
        # draw the center of the circle
        cv2.circle(circle_image, (best[0], best[1]), 2, (0, 0, 255), 3)
        # write coords
        cv2.putText(circle_image, str(best), (best[0], best[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255, 255, 255), 2, cv2.LINE_AA)

    # display the image
    pil_image = Image.fromarray(circle_image)
    pil_image.show()


if __name__ == "__main__":
    pass
