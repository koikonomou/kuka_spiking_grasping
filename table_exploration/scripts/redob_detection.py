import cv2
import rospy
import imutils
import functools
import cv_bridge
import numpy as np
from imutils import paths
from sensor_msgs.msg import Image
from table_exploration.msg import Distance

class Target(object):

    def __init__(self):
        rospy.init_node('object_detection', anonymous = True)
        self.rate = rospy.get_param("rate", 10)
        self.image_topic = rospy.get_param("image_topic", "/kuka/camera1/image_raw")
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.message)
        self.im_pub = rospy.Publisher('/red_image', Image, queue_size=10)
        self.dist_pub = rospy.Publisher('/kuka/box/distance', Distance, queue_size=10)

        self.real_width = 10 #cm
        self.known_distance = 78.8 #cm

    def message(self, msg):
        self.last_msg = msg
        self.stamp = msg.header.stamp

    def mask(self,image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        min_red = np.array([0,230,170]) 
        max_red = np.array([255,255,220]) 
        # min_red = np.array([155,25,0])
        # max_red = np.array([179,255,255])

        mask_r = cv2.inRange(hsv, min_red, max_red) 
        output = cv2.bitwise_and(image, image, mask=mask_r)
        cnts,_ = cv2.findContours(mask_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return cnts 

    # focal length finder function
    def focal_length(self, measured_distance, real_width, width_in_rf_image):
     
        # finding the focal length
        focal_len = (width_in_rf_image * measured_distance) / real_width
        return focal_len
     
    # distance estimation function
    def distance(self, Focal_Length, real_object_width, object_width_in_frame):
     
        distance = (real_object_width * Focal_Length)/object_width_in_frame
     
        # return the distance
        return distance


    def spin(self):
        global object_width
        object_width = 1
        rate = rospy.Rate(self.rate)
        ref_image = cv2.imread("red_gazebo.png")
        ref_image_width = self.mask(ref_image)
        # for c in ref_image_width:
        c = max(ref_image_width, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        object_width = w
            # print("w",w)
        focal_length = self.focal_length(self.known_distance,self.real_width, object_width)
        print("focal_length",focal_length)
        while not rospy.is_shutdown():
            rate.sleep()

            if self.last_msg is None:
                rospy.logwarn_throttle(2, "no image")
                continue
            try:
                image = cv_bridge.imgmsg_to_cv2(self.last_msg, "bgr8")
                marker = self.mask(image)
                max_cnt = max(marker, key = cv2.contourArea)
                # draw a bounding box around the image and display it
                draw_im = cv2.drawContours(image, max_cnt, -1, (0, 255, 0), 3)
                red_image = cv_bridge.cv2_to_imgmsg(draw_im)
                print("Number", len(marker))
                x,y,w,h = cv2.boundingRect(max_cnt)
                width = w
                distance = self.distance(focal_length, self.real_width, width)
                print("Distance", distance)
                self.im_pub.publish(red_image)
                msg = Distance()
                msg.data = distance
                msg.stamp = self.stamp
                self.dist_pub.publish(msg)

            except ValueError:
                rospy.logwarn_throttle(2, "object detection error")

if __name__ == "__main__" :
    node = Target()
    node.spin()