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
        self.rec_image = Image()
        # self.image_inf = [0,self.rec_image]
        self.box_color = (0,255,0)
        self.real_width = 10
        self.known_distance = 78.8

    def on_image(self,msg):
        self.last_msg = msg
        self.stamp = msg.header.stamp

    def draw_rectangles(self, im, detect):
        for (x,y,h,w) in detect:
            rect = cv2.rectangle(im, (x,y),(x+w,y+h), self.box_color, 3)
            rec_image = cv_bridge.cv2_to_imgmsg(rect)
            image_inf=[w,rec_image]
        return image_inf

    def crop_image(self, img, detect):
        for (x,y,h,w) in detect:
            crop_img = img[y:y+h,x:x+w]
            # crop_img = cv_bridge.cv2_to_imgmsg(crop_img)
        return crop_img

    def mask(self,image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        min_red = np.array([0, 0, 50])
        max_red = np.array([50, 50, 255])
        # min_red = np.array([155,25,0])
        # max_red = np.array([179,255,255])

        mask = cv2.inRange(hsv, min_red, max_red) 
        output = cv2.bitwise_and(image, image, mask=mask)
        cnts,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(cnts)
        # c = max(cnts, key = cv2.contourArea)
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
            object_width = w
        return object_width

    def spin(self):
        rate = rospy.Rate(self.rate)
        global object_width
        # object_width = 0.1
        rate = rospy.Rate(self.rate)
        ref_image = cv2.imread("red_gazebo.png")
        ref_image_width = self.mask(ref_image)
        object_width = ref_image_width[1][0]
        focal_length = self.focal_length(self.known_distance,self.real_width, object_width)

        while not rospy.is_shutdown():
            rate.sleep()

            if self.last_msg is None:
                rospy.logwarn_throttle(2, "no image")
                continue
            try:
                image = cv_bridge.imgmsg_to_cv2(self.last_msg, "bgr8")
                # self.last_msg = None
                detections = self.cascade.detectMultiScale(image, 1.2, 5)
                crop_img = self.crop_image( image, detections)
                rec_image = self.draw_rectangles( image , detections)

                # cv2.imshow('Image', crop_img)
                # cv2.waitKey(0) 
                marker = self.mask(crop_img)
                draw_im = cv2.drawContours(crop_img, marker, -1, (0, 255, 0), 2)
                red_obj = cv_bridge.cv2_to_imgmsg(draw_im)
                self.box_pub.publish(red_obj)
                object_width = self.width_in_rfimg(marker)
                distance = self.distance(focal_length,self.real_width, object_width)
                print(distance)
                msg = Distance()
                msg.data = distance
                msg.stamp = self.stamp
                self.dist_pub.publish(msg)
                self.cascade_pub.publish(rec_image[1])


            except ValueError:
                rospy.logwarn_throttle(2, "object detection error")


if __name__ == "__main__":
    node = CascadeBoxDetector()
    node.spin()