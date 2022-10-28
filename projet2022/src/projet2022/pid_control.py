from std_msgs.msg import String
from geometry_msgs.msg import Twist

# ==== PID controller global variables ==== #

# cum_error = 0 # Cumulative error
# de_dt = 0 # change in error
# sleep_rate = 60 # How often we will publish message (useful for derivative and integral controller)


def pid_controller(Kp, x_error, z_error, message):
    """
    Takes linear x error, angular z error, and a rosmsg to modify its values
    -------
    x_error : float or int, linear velocity error
    z_error : float or int, angular velocity error
    message : rosmsg
    -------
    Returns
    message : rosmsg, modified rosmsg with the correct actuator (u_x and u_z) values
    """

    # control command for x (linear velocity)
    #u_x = Kp * error + Ki* cum_error + Kp * de_dt
    u_x = Kp/10 * x_error

    # control command for z (angular velocity)
    # u_z = Kp * error + Ki* cum_error + Kp * de_dt
    u_z = Kp * z_error

    message.linear.x = u_x
    message.angular.z = u_z

    return message