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

# ============ IMAGE TOPIC ============== #

# check if image topic param exists
if rospy.has_param('img_topic_param'):

    IMG_TOPIC_PARAM = rospy.get_param('img_topic_param')
    MSG_TYPE = Image

else:

    IMG_TOPIC_PARAM = '/camera/image'
    MSG_TYPE = Image

# ============= TIME VARIABLES ========== #

TIME_AT_LAST_RED_SEEN = time.time()
TIME_SINCE_LAST_RED = 0
RED_COUNTER = 0

def update_red_counter(img):
    """
    Updates the amount of horizontal red lines we have seen
    """

    red_bool = check_for_red(img)

    global TIME_AT_LAST_RED_SEEN
    global TIME_SINCE_LAST_RED
    global RED_COUNTER

    if red_bool == True:

            # Time elapsed since last time a red line was seen
            TIME_SINCE_LAST_RED = time.time() - TIME_AT_LAST_RED_SEEN

            # if first red line is seen or 5 seconds have passed since last red line seen
            if (RED_COUNTER == 0 or TIME_SINCE_LAST_RED > 5):

                # update values of time at the moment red line was seen
                TIME_AT_LAST_RED_SEEN = time.time()

                # add counter
                RED_COUNTER += 1
                print("NUMBER OF REDS SEEN: ", RED_COUNTER)


# ================================================================ #
# ======================== FUNCTIONS ============================= #
# ================================================================ #

def wall_follow(scan):

    # error variables
    x_error_tab = []
    z_error_tab = []

    # ====== calculate error for all angles in left arc range ====== #
    arc_range = 90

    left_start = 0
    left_end = left_start + arc_range

    for i in range(0,arc_range):

        # lidar distance at i angle
        distance = scan[i]

        # lower the infinite values to measurable value
        if (distance==float('inf')):
            distance = 2
            
        # convert angle to radians
        angle = np.deg2rad(i)

        # horizontal and vertical distance obtained from x and y components of each angle
        horizontal_dist = distance*np.sin(angle)
        vertical_dist = distance*np.cos(angle)

        x_error_tab.append(vertical_dist)
        z_error_tab.append(horizontal_dist)

    # ====== calculate error for all angles in left arc range ====== #
    

    right_start = 359
    right_end = right_start - arc_range

    for i in range(right_start,right_end,-1):
        
        # lidar distance at i angle
        distance = scan[i]

         # lower the infinite values to measurable value
        if (distance==float('inf')):
            distance = 2
            
        # convert angle to radians
        angle = np.deg2rad(i)
        
        # horizontal and vertical distance obtained from x and y components of each angle
        horizontal_dist = distance*np.sin(angle)
        vertical_dist = distance*np.cos(angle)

        x_error_tab.append(vertical_dist)
        z_error_tab.append(horizontal_dist)

    # Change to np array
    x_error_tab = np.array(x_error_tab)
    z_error_tab = np.array(z_error_tab)

    # x_error is the mean value of all vertical distance errors of every angle we recorded
    x_error = np.mean(x_error_tab)

    # z_error is the mean value of all horizontal distance errors of every angle we recorded
    z_error = np.mean(z_error_tab)

    message = Twist()
    
    # PID controller, will make the robot turn towards the direction with the biggest distance
    message = pid_controller(3, x_error, z_error, message)
    vel_pub.publish(message)

def line_follow1(img):

    # ==================== IMAGE PROCESSING ====================== #

    # image variables
    height = img.shape[0]
    width = img.shape[1]

    # Lane side colors
    left_color = "yellow"
    right_color = "white"

    # create the polygon array that frames the lane
    # polygon = create_lane_polygon(img,left_color,right_color)
    polygon = create_lane_polygon(img, left_color, right_color)

    # draw polygon
    lane = draw_lane_polygon(img, polygon)

    # ======================= PID CONTROL ====================== #

    # find center of polygon
    center = moments(polygon)

    # x linear error
    x_error = height - center[1]

    # z angular error
    z_error = int(width/2) - center[0]

    # velocity message to publish
    message = Twist()

    # use pid_controller to modify message values
    message = pid_controller(0.04, x_error, z_error, message)

    # ========== STOPPING CONDITION ============== #
    if (RED_COUNTER >= 2):
        message.linear.x = 0
        message.linear.z = 0

    # publish the message
    vel_pub.publish(message) 

    # display lane mask
    cv.imshow("LANE", lane)
    cv.waitKey(1)

def callback_lidar(msg):

    # lidar scan ranges array
    scan = np.array(msg.ranges)

    # Wall following stopping condition
    if (RED_COUNTER < 1):

        wall_follow(scan)

def callback_img(msg):

    bridge = CvBridge()

    # convert from ros message to cv2 mat with 
    img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # ========== STOPPING CONDITION ============== #
    if (RED_COUNTER >= 1):
        line_follow1(img)

    # update number of red lines we saw
    update_red_counter(img)


def listener():

    # topic and msg defined in global variables
    rospy.Subscriber('/scan', LaserScan, callback_lidar)
    rospy.Subscriber(IMG_TOPIC_PARAM, MSG_TYPE, callback_img)
    rospy.spin()
        
# ================================================================ #
# =========================== MAIN =============================== #
# ================================================================ #

if __name__ == '__main__':

    try:
        
        print("CHALLENGE 2 STARTING...")
        # node name
        rospy.init_node('tunnel', anonymous = True)
        listener()
        
    except rospy.ROSInterruptException:
        pass
