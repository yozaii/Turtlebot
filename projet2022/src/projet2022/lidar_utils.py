import numpy as np
import cv2 as cv
import math

def lidar_points_cloud(img, scan):
    """
    Takes lidar measurements and transforms it into a 2d image
    ---
    img : ndarray, input camera image
    scan : ndarray, lidar ranges value (0 to 359 angle)
    ---
    Returns
    out : ndarray, 2d image of lidar points
    """

    # image variables
    height = img.shape[0]
    width = img.shape[1]

    # robot origin
    center_h = int(height/2)
    center_w = int(width/2)

    # output image
    out = np.zeros((height, width), dtype = 'uint8')

    # output polygon

    for i in range(scan.size):
        
        if (scan[i]<0.3):

            [x,y] = lidar_to_pixel(img, i, scan[i])

            
            out[x, y] = 255
    
    return out

def lidar_to_pixel(img, angle, distance):

        height = img.shape[0]
        width = img.shape[1]

        center_h = height
        center_w = int(width/2)

        scale_factor_x = width
        scale_factor_y = height

        # lidar 0 angle is towards the right in the unit circle
        # we add 90 to the angle and make it negative so that the 0 angle points
        # upwards for the sake of easier integration with the image
        angle = - ((angle + 90) * np.pi / 180)


        x = min(height-1, int(center_h + (distance * np.sin(angle) * scale_factor_x) ))
        x = max(0, x)

        y = min(width-1, int(center_w + (distance * np.cos(angle) * scale_factor_y)))
        y = max(0, y)

        return [x,y]

def lidar_left_top_right(scan):

    # ====== LEFT ======= #

    left = []
 
    for i in range(60, 120):
        left.append(scan[i])

    left = np.array(left)
    left_mean = np.median(left)

    # ======= TOP ======== #

    top = []
 
    for i in range(0, 30):
        top.append(scan[i])

    for i in range(330, 360):
        top.append(scan[i])

    top = np.array(top)
    top_mean = np.median(top)

    # ====== RIGHT ======== #

    right = []
 
    for i in range(240, 300):
        right.append(scan[i])

    right = np.array(right)
    right_mean = np.median(right)

    # ====== RETURN ====== #


    return left_mean, top_mean, right_mean

            