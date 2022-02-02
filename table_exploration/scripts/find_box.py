import os
import json
import time
import math
import rospy
import argparse
import cv_bridge
import cv2
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
# from __future__ import print_function


pose_publisher = None


class CascadeDoxDetector(object):
    def __init__(self):
        rospy.init_node('cascade_box_detector', anonymous = True)
        self.rate = rospy.get_param("rate", 10)
        self.image_topic = rospy.get_param("image_topic", "/kuka/camera1/image_raw")
        self.detector_topic = rospy.get_param("detector_topic", "/table/box/detection")
        self.cascade = cv2.CascadeClassifier('/home/katerina/catkin_ws/src/kuka_spiking_grasping/table_exploration/cascade/cascade.xml')
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.on_image, queue_size = 1)
        self.pub_detection = rospy.Publisher(self.detector_topic, String, queue_size = 1)
        self.pub_a1 = rospy.Publisher('/kuka_kr4r600/joint_a1_position_controller/command', Float64, queue_size=10)
        self.pub_a2 = rospy.Publisher('/kuka_kr4r600/joint_a2_position_controller/command', Float64, queue_size=10)
        self.pub_a3 = rospy.Publisher('/kuka_kr4r600/joint_a3_position_controller/command', Float64, queue_size=10)
        self.pub_a4 = rospy.Publisher('/kuka_kr4r600/joint_a4_position_controller/command', Float64, queue_size=10)
        self.pub_a5 = rospy.Publisher('/kuka_kr4r600/joint_a5_position_controller/command', Float64, queue_size=10)
        self.pub_a6 = rospy.Publisher('/kuka_kr4r600/joint_a6_position_controller/command', Float64, queue_size=10)

    def on_image(self,msg):
        self.last_msg = msg

    # def draw_rectangles(self)

    def spin(self):
        rate = rospy.Rate(self.rate)

        while not rospy.is_shutdown():
            rate.sleep()
            if self.last_msg is None:
                rospy.logwarn_throttle(2, "no image")
                continue

            try:
                t = time.time()
                image = cv_bridge.imgmsg_to_cv2(self.last_msg, "bgr8")
                self.last_msg = None
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                detections = self.cascade.detectMultiScale(gray, 1.2, 5)

                m = String()
                detections_v = [
                    [
                        float(x) / image.shape[0],
                        float(y) / image.shape[1],
                        float(w) / image.shape[0],
                        float(h) / image.shape[1],
                    ] for (x,y,w,h) in detections]
                m.data = json.dumps(detections_v)
                msg_a1 = Float64()
                msg_a2 = Float64()
                msg_a3 = Float64()
                msg_a4 = Float64()
                msg_a5 = Float64()
                msg_a6 = Float64()

                msg_a1.data = -1
                msg_a2.data = -1
                msg_a3.data = 1.4
                msg_a4.data = 0.6
                msg_a5.data = 1
                msg_a6.data = 0.0

                self.pub_detection.publish(m)
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