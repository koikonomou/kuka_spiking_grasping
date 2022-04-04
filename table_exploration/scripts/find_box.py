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

                self.box_pub.publish(rec_image)


            except ValueError:
                rospy.logwarn_throttle(2, "object detection error")


if __name__ == "__main__":
    node = CascadeDoxDetector()
    node.spin()