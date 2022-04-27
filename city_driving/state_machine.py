#!/usr/bin/env python2

from xmlrpc.client import boolean
import rospy
import numpy as np
from safety_controller import SafetyController
from line_follower import LineFollower
from wall_follower import WallFollower
from std_msgs.msg import Float32MultiArray
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import LaserScan, Image
from ackermann_msgs.msg import AckermannDriveStamped

class StateMachine():
    navigation_state = "line_following" #alt wall_follower
    seen_stop_sign = False
    stop_sign_location = None
    car_wash_seen = False
    car_wash_entrance = None
    entered_car_wash = False
    exited_car_wash = False
    stopped = False
    stop_threshold_max = .75
    stop_threshold_min = .50
    current_stop_sign = None
    last_stop_sign = None

    line_follow_drive = None
    wall_follow_drive = None


    def __init__(self):
        self.drive_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/navigation', AckermannDriveStamped)

        #initalize line and wall followers
        self.line_follower = LineFollower()
        self.line_follower = WallFollower()
        self.safety = SafetyController()

        #decide which drive commands to use
        self.input_image_sub = rospy.Subscriber("images", Image, self.state_callback)
        
        #subscribe to line and wall followers
        self.line_follower_sub = rospy.Subscriber("line_follwer", AckermannDriveStamped, self.line_callback)
        self.wall_follower_sub = rospy.Subscriber("wall_follower", AckermannDriveStamped, self.wall_callback)

        #subscribe to stop signs 
        self.stop_sign_sub = rospy.Subscriber("stop_sign_bbox", Float32MultiArray, self.stop_sign_callback)
    
    def state_callback(self, msg):
        # stop sign detection is back box ta code
        # TODO: Change to use TA code
        #get uv pixel -> homography -> compute distance


        self.seen_stop_sign = staff_find_stop_sign("boolean")

        if self.seen_stop_sign and not self.stopped:
            self.stop_sign_location = staff_find_stop_sign("location")
            if self.need_stop(self.stop_sign_location):
                self.stopped = True
                ackermann_drive = AckermannDriveStamped()
                ackermann_drive.header.stamp = rospy.Time.now()
                ackermann_drive.header.frame_id = "map"
                ackermann_drive.drive.speed = 0
                self.drive_pub.publish(ackermann_drive)

            else:
                self.drive_pub.publish(self.line_follow_drive)


        #make function to determine when to enter the car wash
        elif self.car_wash_seen and not self.exited_car_wash:
            # consider leaving the carwash and refinding the line
            # toggle safety controller to enter car wash
            pass #TODO: fill in when adding car wash short cut

        else: #default line following case
            self.drive_pub.publish(self.line_follow_drive)


        pass

    def line_callback(self, msg):
        self.line_follow_drive = msg

    def wall_callback(self, msg):
        self.wall_follow_drive = msg

    def stop_sign_callback(self, msg):
        self.wall_follow_drive = msg

    def need_stop(self, pixel):
        # pixel is stop sign location in x y
        if np.sqrt(pixel.x**2+pixel.y**2) >= self.stop_threshold_min and np.sqrt(pixel.x**2+pixel.y**2) <= self.stop_threshold_max:
            if not self.stopped:
                return True
        
        elif self.stopped and np.sqrt(pixel.x**2+pixel.y**2) <= self.stop_threshold_min:
            self.stopped = False
            return False

        return False


    





if __name__ == '__main__':
    rospy.init_node('state_machine')
    state_machine = StateMachine()
    rospy.spin()
