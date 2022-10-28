import numpy as np
import time
import rospy
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError


from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import message_filters # synchronization


from projet2022.img_utils import *
from projet2022.pid_control import pid_controller
from projet2022.lidar_utils import *

# ============================================================== #
# ====================== GLOBAL VARIABLES ====================== #
# ============================================================== #

# ============== PUBLISHER ============== #

# name of velocity command topic
if rospy.has_param('vel_cmd_topic'):

    topic_param = rospy.get_param('vel_cmd_topic')

else :
    
    topic_param = '/cmd_vel'

# velocity publisher object
vel_pub = rospy.Publisher(topic_param, Twist, queue_size=10)


# ================================================================ #
# ======================== FUNCTIONS ============================= #
# ================================================================ #

def callback(msg):

    # lidar scan ranges array
    scan = np.array(msg.ranges)

    # error variables
    x_error_tab = []
    z_error_tab = []

    # ====== error for all angles in left arc range ====== #
    arc_range = 70

    left_start = 0
    left_end = left_start + arc_range

    # we don't record values above max_distance to prevent
    # large distances outside tunnel dominating the error values
    max_distance = 2

    for i in range(0,arc_range):

        distance = scan[i]

        if (distance> max_distance):
            distance = max_distance
            
        # convert angle to radians
        angle = np.deg2rad(i)



        horizontal_dist = (distance)*np.sin(angle)
        vertical_dist = (distance)*np.cos(angle)

        x_error_tab.append(vertical_dist)
        z_error_tab.append(horizontal_dist)

    # ====== error for all angles in left arc range ====== #

    right_start = 359
    right_end = right_start - arc_range

    for i in range(right_start,right_end,-1):

        distance = scan[i]

        if (distance> max_distance):
            distance = max_distance
            
        # convert angle to radians
        angle = np.deg2rad(i)

        horizontal_dist = (distance)*np.sin(angle)
        vertical_dist = (distance)*np.cos(angle)

        x_error_tab.append(vertical_dist)
        z_error_tab.append(horizontal_dist)


    x_error_tab = np.array(x_error_tab)
    z_error_tab = np.array(z_error_tab)

    # x_error
    x_error = np.mean(x_error_tab)

    # z_error
    z_error = np.mean(z_error_tab)*4

    message = Twist()
    
    message = pid_controller(1, x_error, z_error, message)
    vel_pub.publish(message)

    # print("TOP MEDIAN: ", top_median)
    # print( "LEFT MEDIAN: ", left_median, "RIGHT MEDIAN:", right_median)
    # print("TOP RIGHT MEDIAN", topr_median, "TOP LEFT MEDIAN", topl_median)
    print("x",x_error)
    print("z",z_error)


def listener():

    # topic and msg defined in global variables
    rospy.Subscriber('/scan', LaserScan, callback)
    rospy.spin()
        
# ================================================================ #
# =========================== MAIN =============================== #
# ================================================================ #

if __name__ == '__main__':

    try:
        # node name
        rospy.init_node('tunnel', anonymous = True)
        
        listener()
        
    except rospy.ROSInterruptException:
        pass
