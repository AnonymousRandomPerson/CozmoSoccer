#!/usr/bin/env python3

import cv2
import sys
import copy

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit('install Pillow to run this code')


def find_ball(opencv_image, debug=False):
    """Find the ball in an image.

        Arguments:
        opencv_image -- the image
        debug -- an optional argument which can be used to control whether
                debugging information is displayed.

        Returns [x, y, radius] of the ball, and [0,0,0] or None if no ball is found.
    """

    ball = None

    # INSERT SOLUTION
    #opencv_image = cv2.blur(opencv_image,(7, 7))

    # opencv_image = cv2.normalize(opencv_image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # mask = cv2.inRange(opencv_image, 0, 150)
    # opencv_image = cv2.bitwise_and(opencv_image, opencv_image, mask = mask)
    # cv2.imshow('img', opencv_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ball_keypoints = cv2.HoughCircles(
        opencv_image, cv2.HOUGH_GRADIENT, 1.3, 7, param1=70, param2=20)

    if ball_keypoints is None:
        return None

    for keypoint in ball_keypoints:

        circ_x = keypoint[0][0]
        circ_y = keypoint[0][1]
        circ_rad = keypoint[0][2]

        ## INTENSITY CHECK ##
        # @TODO Could be too strict if ball gets reflected with light.
        INTENSITY_THRESHOLD = 50
        INTENSITY_MIN_RATE = .75
        count = 0
        tot_pixels = 0
        # Fit a square within detected circle.
        bound = np.floor(np.sqrt(np.square(2 * circ_rad) / 2) / 2)
        bound_start_x = circ_x - bound
        bound_start_y = circ_y - bound

        # Fix bound if negative
        if bound_start_x < 0:
            bound_start_x = 0
        if bound_start_y < 0:
            bound_start_y = 0

        bound_end_x = (bound_start_x + bound * 2)
        bound_end_y = (bound_start_y + bound * 2)

        for y in range(int(bound_start_y), int(bound_end_y)):
            for x in range(int(bound_start_x), int(bound_end_x)):
                # Prevent out of bounds
                if y >= opencv_image.shape[0] or x >= opencv_image.shape[1]:
                    continue
                tot_pixels += 1
                if opencv_image[y, x] < INTENSITY_THRESHOLD:
                    count += 1

        if count > INTENSITY_MIN_RATE * tot_pixels:
            if debug:
                circles = []
                circles.append([circ_x, circ_y, circ_rad])
                display_circles(opencv_image, circles)

            return [circ_x, circ_y, circ_rad]

    return None


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
