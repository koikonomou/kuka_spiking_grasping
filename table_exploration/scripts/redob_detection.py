import cv2
import rospy
import imutils
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

    def message(self, msg):
        self.last_msg = msg

    def mask(self,image):

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        min_red = np.array([0,230,170]) 
        max_red = np.array([255,255,220]) 

        mask_r = cv2.inRange(hsv, min_red, max_red) 
        output = cv2.bitwise_and(image, image, mask=mask_r)
        cnts,_ = cv2.findContours(mask_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return cnts, output


    def spin(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            rate.sleep()

            if self.last_msg is None:
                rospy.logwarn_throttle(2, "no image")
                continue
            try:
                image = cv_bridge.imgmsg_to_cv2(self.last_msg, "bgr8")
                marker = self.mask(image)[0]
                # draw a bounding box around the image and display it
                draw_im = cv2.drawContours(image, marker, -1, (0, 255, 0), 2)
                red_image = cv_bridge.cv2_to_imgmsg(draw_im)
                self.im_pub.publish(red_image)

            except ValueError:
                rospy.logwarn_throttle(2, "object detection error")

if __name__ == "__main__" :
    node = Target()
    node.spin()