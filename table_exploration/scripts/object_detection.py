import cv2
import rospy
import imutils
import functools
import cv_bridge
import numpy as np
from imutils import paths
from sensor_msgs.msg import Image
from table_exploration.msg import Distance



class CascadeBoxDetector(object):

    def __init__(self):
        rospy.init_node('cascade_box_detector', anonymous = True)
        self.rate = rospy.get_param("rate", 10)
        self.image_topic = rospy.get_param("image_topic", "/kuka/camera1/image_raw")
        self.detector_topic = rospy.get_param("detector_topic", "/table/box/detection")
        self.cascade = cv2.CascadeClassifier('/home/katerina/catkin_ws/src/kuka_spiking_grasping/table_exploration/cascade/cascade.xml')
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.on_image, queue_size = 1)
        self.dist_pub = rospy.Publisher('/kuka/box/distance', Distance, queue_size=10)
        self.box_pub = rospy.Publisher(self.detector_topic, Image, queue_size=1)
        self.cascade_pub = rospy.Publisher('/kuka/cascade/detection', Image, queue_size=1)
        self.box_color = (0,255,0)
        self.real_width = 0.10
        self.known_distance = 1.6
        self.rec_image = Image()
        self.image_rect = [0,self.rec_image]
        self.crop_img = Image()
        self.object_width = 0
        self.detections = np.array([])


    def on_image(self,msg):
        self.last_msg = msg
        self.stamp = msg.header.stamp

    def draw_rectangles(self, im, detect):
        for (x,y,h,w) in detect:
            rect = cv2.rectangle(im, (x,y),(x+w,y+h), self.box_color, 3)
            self.image_rect = cv_bridge.cv2_to_imgmsg(rect)
            self.image_rect=[w,self.image_rect]
        return self.image_rect

    def crop_image(self, img, detect):
        for (x,y,h,w) in detect:
            self.crop_img = img[y:y+h,x:x+w]
        return self.crop_img

    def mask(self,image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        min_red = np.array([0, 0, 50])
        max_red = np.array([50, 50, 255])

        mask = cv2.inRange(hsv, min_red, max_red) 
        output = cv2.bitwise_and(image, image, mask=mask)
        cnts,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return cnts

    def focal_length(self, measured_distance, real_width, width_in_rf_image):
        focal_len = (width_in_rf_image * measured_distance) / real_width
        return focal_len

    def distance(self, focal_len, real_object_width, object_width_in_frame):
        distance = (real_object_width * focal_len)/object_width_in_frame
        return distance

    def width_in_rfimg(self, marker):
        for c in marker:
            x,y,w,h = cv2.boundingRect(c)
            self.object_width = w
        return self.object_width

    def find_detections(self, image):
        self.detections = self.cascade.detectMultiScale(image, 1.2, 5)
        return self.detections

    def spin(self):
        rate = rospy.Rate(self.rate)

        ref_image = cv2.imread("red_gazebo.png")
        ref_image_width = self.mask(ref_image)
        self.object_width = self.width_in_rfimg(ref_image_width)

        focal_length = self.focal_length(self.known_distance,self.real_width, self.object_width)
        while not rospy.is_shutdown():
            rate.sleep()

            if self.last_msg is None:
                rospy.logwarn_throttle(2, "no image")
                continue
            try:
                image = cv_bridge.imgmsg_to_cv2(self.last_msg, "bgr8")
                detections = self.find_detections(image)
                if not len(detections):
                    rospy.logwarn_throttle(2, "Cascade failed to detect object ")
                    msg = Distance()
                    msg.data = -10
                    msg.stamp = self.stamp
                    self.dist_pub.publish(msg)
                else:
                    crop_img = self.crop_image( image, detections)
                    self.rec_image = self.draw_rectangles( image , detections)
                    # cv2.imshow('Image', crop_img)
                    # cv2.waitKey(0) 
                    marker = self.mask(crop_img)

                    draw_im = cv2.drawContours(crop_img, marker, -1, self.box_color, 2)
                    red_obj = cv_bridge.cv2_to_imgmsg(draw_im)

                    self.box_pub.publish(red_obj)
                    object_width = self.width_in_rfimg(marker)
                    distance_est = self.distance(focal_length, self.real_width, object_width)
                    # print (distance_est)

                    msg = Distance()
                    msg.data = distance_est
                    msg.stamp = self.stamp
                    self.dist_pub.publish(msg)
                    self.cascade_pub.publish(self.rec_image[1])


            except ValueError:
                rospy.logwarn_throttle(2, "object detection error")


if __name__ == "__main__":
    node = CascadeBoxDetector()
    node.spin()