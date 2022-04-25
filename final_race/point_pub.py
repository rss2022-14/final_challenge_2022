#!/usr/bin/env python

import rospy
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker

#The following collection of pixel locations and corresponding relative
#ground plane locations are used to compute our homography matrix
######################################################

# note: may still need tuning
PTS_IMAGE_PLANE = [[458, 248], [317, 251], [302, 300], [505, 295]]
PTS_GROUND_PLANE = [[25.5, -5.5], [25.5, 5.5], [17, 5.5], [17, -5.5]]
METERS_PER_INCH = 0.0254

class HomographyTransformer:
    def __init__(self):
        self.hough_sub = rospy.Subscriber('/hough_out', msg, self.hough_cb)
        self.point_pub = rospy.Publisher('/pursuit_point', Point, queue_size=10)

        self.marker_pub = rospy.Publisher('/pursuit_viz', Marker, queue_size=1)

        self.look_ahead = 0.3 #TO-DO: set this in rosparam file
        
        np_pts_ground = np.array(PTS_GROUND_PLANE)
        np_pts_ground = np_pts_ground * METERS_PER_INCH
        np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

        np_pts_image = np.array(PTS_IMAGE_PLANE)
        np_pts_image = np_pts_image * 1.0
        np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

        self.uv2xy, err = cv2.findHomography(np_pts_image, np_pts_ground)
        self.xy2uv = np.linalg.inv(self.uv2xy)

        # precompute v of point directly in front at look ahead
        straight_xy = np.array([0, self.look_ahead, 1]).T
        self.vq = np.dot(self.xy2uv, straight_xy)[1, 0]

    def hough_cb(self, msg):
        '''
        input (np array shape 2x4): list of two lines, each specified by two end points [u0, v0, u1, v1]
        output: pursuit point (at a look ahead distance) published in world frame
        '''
        # Find pursuit point in pixel frame
        l_m = (msg[0,3] - msg[0,1])/(msg[0,2] - msg[0,0])
        left_u = (self.vq - msg[0, 1])/l_m
        r_m = (msg[1,3] - msg[1,1])/(msg[1,2] - msg[1,0])
        right_u = (self.vq - msg[1, 1])/r_m

        uq = (right_u - left_u)/2
        pursuit_uv = np.array([uq, self.vq, 1]).T
        
        # Transform pursuit point to robot frame
        pursuit_xy = (np.dot(self.uv2xy, pursuit_uv)[:2, 0]).T

        # Publish
        self.point_pub.publish(pursuit_xy)
        self.marker_pub.publish()
        print('pursuit point published')


if __name__ == "__main__":
    rospy.init_node('homography_transformer')
    homography_transformer = HomographyTransformer()
    rospy.spin()
