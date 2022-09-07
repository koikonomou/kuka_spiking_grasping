# ! /usr/bin/env python3
import math
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import LaserScan
from table_exploration.msg import Collision


class Distance(object):
    def __init__(self):
        rospy.init_node('scan_collision', anonymous=True)
        self.scan_cb_init = False
        self.actual_dist_cb_init = False
        self.rate = rospy.get_param("rate",10)
        self.pub = rospy.Publisher('/kuka/collision', Float64, queue_size=1)
        self.sub = rospy.Subscriber('/kuka/laser/scan', LaserScan, self.scan_cb, queue_size=10)
        self.dist_sub = rospy.Subscriber('/collision_detection', Collision, self.dist_cb, queue_size=10)
        while not self.scan_cb_init:
            continue
        while not self.actual_dist_cb_init:
            continue
        rospy.loginfo("Finish Subscriber Init...")

    def scan_cb(self,msg):
        if self.scan_cb_init is False :
            self.scan_cb_init = True
        dist = min(msg.ranges[288:431])
        self.dist = dist

    def dist_cb(self,msg):
        if self.actual_dist_cb_init is False:
            self.actual_dist_cb_init = True
        actual_dist = msg.goal_dist
        self.actual_dist = actual_dist


    def spin(self):
        rate = rospy.Rate(self.rate)

        while not rospy.is_shutdown():
            rate.sleep()
            try:
                distance = self.dist
                if math.isnan(distance) == True :
                    distance = 20
                elif math.isinf(distance) == True:
                    distance = 20
                elif distance > 0.90 :
                    distance = 20 

                test= Float64()
                test.data = distance 
                self.pub.publish(test)
            except ValueError:
                rospy.logwarn_throttle(2, "object detection error")


if __name__=='__main__':
    node = Distance()
    node.spin()