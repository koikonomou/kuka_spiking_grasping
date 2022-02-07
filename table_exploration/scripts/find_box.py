import os
import cv2
import json
import time
import math
import rospy
import argparse
import cv_bridge
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from scipy.spatial import distance as dist


class CascadeDoxDetector(object):

    def __init__(self):
        rospy.init_node('cascade_box_detector', anonymous = True)
        self.rate = rospy.get_param("rate", 10)
        self.image_topic = rospy.get_param("image_topic", "/kuka/camera1/image_raw")
        self.detector_topic = rospy.get_param("detector_topic", "/table/box/detection")
        self.cascade = cv2.CascadeClassifier('/home/katerina/catkin_ws/src/kuka_spiking_grasping/table_exploration/cascade/cascade.xml')
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.on_image, queue_size = 1)
        self.pub_a1 = rospy.Publisher('/kuka_kr4r600/joint_a1_position_controller/command', Float64, queue_size=10)
        self.pub_a2 = rospy.Publisher('/kuka_kr4r600/joint_a2_position_controller/command', Float64, queue_size=10)
        self.pub_a3 = rospy.Publisher('/kuka_kr4r600/joint_a3_position_controller/command', Float64, queue_size=10)
        self.pub_a4 = rospy.Publisher('/kuka_kr4r600/joint_a4_position_controller/command', Float64, queue_size=10)
        self.pub_a5 = rospy.Publisher('/kuka_kr4r600/joint_a5_position_controller/command', Float64, queue_size=10)
        self.pub_a6 = rospy.Publisher('/kuka_kr4r600/joint_a6_position_controller/command', Float64, queue_size=10)
        self.box_pub = rospy.Publisher('/table/detected_box', Image, queue_size=1)
        self.rec_image = Image()

    def on_image(self,msg):
        self.last_msg = msg

    def draw_rectangles(self, im, detect):
        # self.im = im
        # self.detect = detect
        for (x,y,w,h) in detect:
            rect = cv2.rectangle(im, (x,y),(x+w,y+h), (0,255,0),3)
            self.rec_image = cv_bridge.cv2_to_imgmsg(rect)
        return self.rec_image

    def spin(self):
        rate = rospy.Rate(self.rate)

        while not rospy.is_shutdown():
            rate.sleep()

            if self.last_msg is None:
                rospy.logwarn_throttle(2, "no image")
                continue
            try:
                t = time.time()
                self.image = cv_bridge.imgmsg_to_cv2(self.last_msg, "bgr8")
                self.last_msg = None
                self.detections = self.cascade.detectMultiScale(self.image, 1.2, 5)

                rec_image = self.draw_rectangles( self.image , self.detections)

                msg_a1 = Float64()
                msg_a2 = Float64()
                msg_a3 = Float64()
                msg_a4 = Float64()
                msg_a5 = Float64()
                msg_a6 = Float64()


###### Robot arm full position (a1= 0, a2=-90, a3= 0, a4= 0, a5= 0 , a6= 0) ######
###### Robot joint limits in rads (a1= , a2= , a3= , a4= ,a5= 2.09, a6=  ) ######

                msg_a1.data = 0 * math.pi/180
                msg_a2.data = -95 * math.pi/180
                msg_a3.data = 50 * math.pi/180
                msg_a4.data = 0 * math.pi/180
                msg_a5.data = 100 * math.pi/180
                msg_a6.data = 0 * math.pi/180

                self.box_pub.publish(rec_image)
                self.pub_a1.publish(msg_a1)
                self.pub_a2.publish(msg_a2)
                self.pub_a3.publish(msg_a3)
                self.pub_a4.publish(msg_a4)
                self.pub_a5.publish(msg_a5)
                self.pub_a6.publish(msg_a6)

            except ValueError:
                rospy.logwarn_throttle(2, "object detection error")


if __name__ == "__main__":
    node = CascadeDoxDetector()
    node.spin()