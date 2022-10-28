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

# ============ IMAGE TOPIC ============== #

# check if image topic param exists
if rospy.has_param('img_topic_param'):

    IMG_TOPIC_PARAM = rospy.get_param('img_topic_param')
    MSG_TYPE = CompressedImage

else :
    IMG_TOPIC_PARAM = '/rapiscam_node/image/compressed'
    MSG_TYPE = CompressedImage

# ============== PUBLISHER ============== #

# name of velocity command topic
if rospy.has_param('vel_cmd_topic'):

    topic_param = rospy.get_param('vel_cmd_topic')

else :
    
    topic_param = '/cmd_vel'

# velocity publisher object
vel_pub = rospy.Publisher(topic_param, Twist, queue_size=10)

# ============= EMERGENCY STOP ========== #

STOP = False

# ================================================================ #
# ======================== FUNCTIONS ============================= #
# ================================================================ #

def onMouse(event, y, x, flags, prm) :
		if event == cv.EVENT_LBUTTONDOWN :
                
			if type(prm) != type(None) :

				b = prm[x, y, 0]
				g = prm[x, y, 1]
				r = prm[x, y, 2]

				print(f"[{x}, {y}] : BGR({b}, {g}, {r})")
			else :
				print(f"[{x}, {y}]")           

def line_follow1(img):

    # ==================== IMAGE PROCESSING ====================== #

    # image variables
    height = img.shape[0]
    width = img.shape[1]

    # Cover bottom so we can't see wheels
    img[int(9*height/10):-1, :, :] = 0


    # Lane side colors
    left_color = "green"
    right_color = "red"

    # red_img_channel
    red_img = img[:,:,0]

    # create the polygon array that frames the lane
    # polygon = create_lane_polygon(img,left_color,right_color)
    polygon = create_lane_polygon_real(img, right_color)

    # draw polygon
    lane = draw_lane_polygon(img, polygon)

    # ======================= PID CONTROL ====================== #

    # find center of polygon
    center = moments(polygon)

    # x linear error
    x_error = (height - center[1])/2

    # z angular error
    z_error = int(6*width/10) - center[0]

    # velocity message to publish
    message = Twist()

    # use pid_controller to modify message values
    message = pid_controller(0.006, x_error, z_error*2, message)

    # publish the message if there is no call to emergency stop
    if not STOP:
        pass
        vel_pub.publish(message) 

    else :
        message.linear.x = 0
        message.linear.z = 0


    # display lane mask
    cv.imshow("LANE", lane)
    cv.waitKey(1)

def callback(msg):

    np_arr = np.frombuffer(msg.data, np.uint8)
    img = cv.imdecode(np_arr, cv.IMREAD_COLOR)


    # image variables
    height = img.shape[0]
    width = img.shape[1]

    # Cover bottom
    img[int(9*height/10):-1, :, :] = 0

    line_follow1(img)
        
    # update number of red lines we saw
    # update_red_counter(img)

def lidar_callback(msg):

    # 360 lidar angle values
    scan = np.array(msg.ranges)

    # 10 angle range in front of ro
    top_mean = (np.mean(scan[0:5]) + np.mean(scan[345:359]))/2


    # Emergency stop if in front too close
    global STOP
    if top_mean < 0.1:
        STOP = True

    else:
        STOP = False


def listener():

    # topic and msg defined in global variables
    rospy.Subscriber(IMG_TOPIC_PARAM, MSG_TYPE, callback)  
    rospy.Subscriber('/scan', LaserScan, lidar_callback)
    
    rospy.spin()

# ================================================================ #
# =========================== MAIN =============================== #
# ================================================================ #

if __name__ == '__main__':

    try:
        # node name
        rospy.init_node('line_follow', anonymous = True)
        listener()
        # listener2()

    except rospy.ROSInterruptException:
        pass
