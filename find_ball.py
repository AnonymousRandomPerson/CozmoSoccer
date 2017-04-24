#!/usr/bin/env python3

import cv2
import sys
import copy
import time

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit('install Pillow to run this code')


def find_ball(robot, opencv_image, mask, debug=False):
    """Find the ball in an image.

        Arguments:
        opencv_image -- the image
        debug -- an optional argument which can be used to control whether
                debugging information is displayed.

        Returns [x, y, radius] of the ball, and [0,0,0] or None if no ball is found.
    """
    #opencv_image = cv2.blur(opencv_image,(7, 7))

    # opencv_image = cv2.normalize(opencv_image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # mask = cv2.inRange(opencv_image, 0, 150)
    # opencv_image = cv2.bitwise_and(opencv_image, opencv_image, mask = mask)
    # cv2.imshow('img', opencv_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # @TODO test this
    canny_image = cv2.Canny(mask, 0, 50, apertureSize=5)
    if debug:
        cv2.waitKey(1)
        #cv2.imshow('canny', canny_image)
        cv2.imshow('mask', mask)
    IMG_HEIGHT = opencv_image.shape[0]
    IMG_WIDTH = opencv_image.shape[1]

    b, contours, hierarchy = cv2.findContours(
        canny_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Keep top 5
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    if contours is None:
        return None

    center = None

    for cnt in contours:
        ((circ_x, circ_y), circ_rad) = cv2.minEnclosingCircle(cnt)

        # @TODO
        # Ball cannot be above the arena (higher than half of camera's FOV)
        if circ_y < IMG_HEIGHT / 2:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if circ_rad > 16 and circ_rad < 100:
            cv2.circle(opencv_image, (int(circ_x), int(circ_y)),
                       int(circ_rad), (0, 255, 255), 2)
            #cv2.circle(opencv_image, center, 5, (0, 0, 255), -1)
            ## INTENSITY CHECK ##
            # @TODO Could be too strict if ball gets reflected with light.
            INTENSITY_MIN_RATE = .55
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
                    if y >= IMG_HEIGHT or x >= IMG_WIDTH:
                        continue
                    tot_pixels += 1
                    pixel = opencv_image[y, x]
                    if pixel[0] < 86 and pixel[1] < 101 and pixel[2] < 86 and pixel[0] > 24 and pixel[1] > 39 and pixel[2] > 24:
                        count += 1
            if count > INTENSITY_MIN_RATE * tot_pixels:
                if debug:
                    cv2.circle(opencv_image, center, 5, (255, 0, 255), -1)
                    cv2.imshow('Ball', opencv_image)
                return [circ_x, circ_y, circ_rad]
    if debug:
        cv2.imshow('Ball', opencv_image)

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
