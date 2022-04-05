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
from table_exploration.msg import Distance


class CascadeDoxDetector(object):

    def __init__(self):
        rospy.init_node('cascade_box_detector', anonymous = True)
        self.rate = rospy.get_param("rate", 10)
        self.image_topic = rospy.get_param("image_topic", "/kuka/camera1/image_raw")
        self.detector_topic = rospy.get_param("detector_topic", "/table/box/detection")
        self.cascade = cv2.CascadeClassifier('/home/katerina/catkin_ws/src/kuka_spiking_grasping/table_exploration/cascade/cascade.xml')
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.on_image, queue_size = 1)
        self.dist_pub = rospy.Publisher('/kuka/box/distance', Distance, queue_size=10)
        self.box_pub = rospy.Publisher('/table/detected_box', Image, queue_size=1)
        self.rec_image = Image()
        self.image_inf = [0,self.rec_image]
        self.white = (255,255,255)
        self.green = (0,255,0)
        self.known_width = 0.10
        self.knowDistance = 24.0
        self.fonts = cv2.FONT_HERSHEY_COMPLEX

    def focal_length(self, measured_distance, real_width, width_in_rf_image):
        """
        This Function Calculate the Focal Length(distance between lens to CMOS sensor), it is simple constant we can find by using
        MEASURED_DISTACE, REAL_WIDTH(Actual width of object) and WIDTH_OF_OBJECT_IN_IMAGE
        :param1 Measure_Distance(int): It is distance measured from object to the Camera while Capturing Reference image
        :param2 Real_Width(int): It is Actual width of object, in real world (like My face width is = 14.3 centimeters)
        :param3 Width_In_Image(int): It is object width in the frame /image in our case in the reference image(found by Face detector)
        :retrun focal_length(Float):"""
        focal_length_value = (width_in_rf_image * measured_distance) / real_width
        return focal_length_value

    def distance_calc(self, focal_length, real_box_width, box_width_in_frame):
        """
        This Function simply Estimates the distance between object and camera using arguments(focal_length, Actual_object_width, Object_width_in_the_image)
        :param1 focal_length(float): return by the focal_length_Finder function
        :param2 Real_Width(int): It is Actual width of object, in real world (like My face width is = 5.7 Inches)
        :param3 object_Width_Frame(int): width of object in the image(frame in our case, using Video feed)
        :return Distance(float) : distance Estimated
        """
        distance = (real_box_width * focal_length) / box_width_in_frame
        return distance

    def on_image(self,msg):
        self.last_msg = msg
        self.stamp = msg.header.stamp

    def draw_rectangles(self, im, detect):
        for (x,y,h,w) in detect:
            rect = cv2.rectangle(im, (x,y),(x+w,y+h), self.green,3)
            self.rec_image = cv_bridge.cv2_to_imgmsg(rect)
            self.image_inf=[w,self.rec_image]
        return self.image_inf

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

                focal_length = self.focal_length(self.knowDistance,self.known_width, 0.05)
                width_in_frame = rec_image[0]
                if width_in_frame != 0 :
                    self.distance = self.distance_calc(focal_length,self.known_width,width_in_frame)
                    # cv2.putText(self.image, f"Distance = {round(self.distance,2)} CM", (50, 50), self.fonts, 1, (self.white), 2)
                # print(self.distance)
                self.box_pub.publish(rec_image[1])
                msg = Distance()
                msg.data = self.distance
                msg.stamp = self.stamp
                self.dist_pub.publish(msg)
                # if cv2.waitKey(1) == ord("q"):
                #     break

            except ValueError:
                rospy.logwarn_throttle(2, "object detection error")


if __name__ == "__main__":
    node = CascadeDoxDetector()
    node.spin()