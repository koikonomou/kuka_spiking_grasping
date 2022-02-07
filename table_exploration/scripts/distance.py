from imutils import paths
import numpy as np
import cv2
import rospy
import imutils
import cv_bridge
from sensor_msgs.msg import Image
from table_exploration.msg import Distance



class DistaceToObject(object):

    def __init__(self):
        rospy.init_node('distance_estimation', anonymous = True)
        self.rate = rospy.get_param("rate", 10)
        self.image_topic = rospy.get_param("image_topic", "/kuka/camera1/image_raw")
        self.detector_topic = rospy.get_param("detector_topic", "/table/box/detection")
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.distance_callback, queue_size = 1)
        self.dist_pub = rospy.Publisher('/kuka/box/distance', Distance, queue_size=10)
        self.dist_im_pub = rospy.Publisher('/kuka/box/image/distance', Image, queue_size=1)
        self.known_width = 0.10
        self.knowDistance = 24.0

    def distance_callback(self, msg):
        self.last_msg = msg
        self.stamp = msg.header.stamp

    def distance_calc(self, im):
        # convert the image to grayscale, blur it, and detect edges
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 35, 125)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key = cv2.contourArea)
        # compute the bounding box 
        return cv2.minAreaRect(c)

    def distance_to_camera(self, knownWidth, focalLength, perWidth):
        # compute and return the distance from the maker to the camera
        return (knownWidth * focalLength) / perWidth

    def spin(self):

        rate = rospy.Rate(self.rate)

        while not rospy.is_shutdown():
            rate.sleep()

            if self.last_msg is None:
                rospy.logwarn_throttle(2, "no image")
                continue
            try:
                image = cv_bridge.imgmsg_to_cv2(self.last_msg, "bgr8")
                marker = self.distance_calc(image)
                focalLength = (marker[1][0] * self.knowDistance) / self.known_width
                inches = self.distance_to_camera(self.known_width, focalLength, marker[1][0])

                msg = Distance()
                msg.data = inches
                msg.stamp = self.stamp
                self.dist_pub.publish(msg)


                # draw a bounding box around the image and display it
                box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
                box = np.int0(box)
                draw_im = cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

                rec_image = cv_bridge.cv2_to_imgmsg(draw_im)
                self.dist_im_pub.publish(rec_image)

            except ValueError:
                rospy.logwarn_throttle(2, "object detection error")

if __name__ == "__main__" :
    node = DistaceToObject()
    node.spin()