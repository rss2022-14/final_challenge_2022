#!/usr/bin/env python2

import rospy
import numpy as np
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class StateMachine():
    navigation_state = "line_following" #alt wall_follower
    seen_stop_sign = False
    stop_sign_location = None
    car_wash_seen = False
    car_wash_entrance = None
    in_car_wash = False



    def __init__(self):
        self.drive_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/navigation', AckermannDriveStamped)

        #initalize line and wall followers
        self.line_follower = LineFollower()
        self.line_follower = WallFollower()




    # def callback(scan):
    # pub = rospy.Publisher(rospy.get_param('publish_topic','open_space'), OpenSpace)
    # distance = max(scan.ranges)
    # index = scan.ranges.index(distance)
    # angle = (index + 1) * scan.angle_increment + scan.angle_min if (index != len(scan.ranges) - 1) else scan.angle_max

    # out = OpenSpace()
    # out.angle = angle
    # out.distance = distance

    # rospy.loginfo(out)
    # pub.publish(out)

    # def listener():
    # rospy.init_node('open_space_publisher')
    # rospy.Subscriber(self.DRIVE_TOPIC, AckermannDriveStamped, callback)

    # rospy.spin()

if __name__ == '__main__':
    rospy.init_node('saftey_controller')
    safety_controller = SafetyController()
    rospy.spin()
