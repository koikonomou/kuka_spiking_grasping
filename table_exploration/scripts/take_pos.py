import time
import math
import rospy
import numpy as np
from std_msgs.msg import Float64


class Position(object):

    def __init__(self):
        rospy.init_node('new_init_pose', anonymous = True)
        self.rate = rospy.get_param("rate", 10)
        self.ja1 = rospy.get_param("joint_a1", 0 * math.pi/180)
        self.ja2 = rospy.get_param("joint_a2", -48 * math.pi/180)
        self.ja3 = rospy.get_param("joint_a3", 80 * math.pi/180)
        self.ja4 = rospy.get_param("joint_a4", 0 * math.pi/180)
        self.ja5 = rospy.get_param("joint_a5", 35 * math.pi/180)
        self.ja6 = rospy.get_param("joint_a6", 0 * math.pi/180)
        self.pub_a1 = rospy.Publisher('/kuka_kr4r600/joint_a1_position_controller/command', Float64, queue_size=10)
        self.pub_a2 = rospy.Publisher('/kuka_kr4r600/joint_a2_position_controller/command', Float64, queue_size=10)
        self.pub_a3 = rospy.Publisher('/kuka_kr4r600/joint_a3_position_controller/command', Float64, queue_size=10)
        self.pub_a4 = rospy.Publisher('/kuka_kr4r600/joint_a4_position_controller/command', Float64, queue_size=10)
        self.pub_a5 = rospy.Publisher('/kuka_kr4r600/joint_a5_position_controller/command', Float64, queue_size=10)
        self.pub_a6 = rospy.Publisher('/kuka_kr4r600/joint_a6_position_controller/command', Float64, queue_size=10)

    def spin(self):
        rate = rospy.Rate(self.rate)

        while not rospy.is_shutdown():
            rate.sleep()

            msg_a1 = Float64()
            msg_a2 = Float64()
            msg_a3 = Float64()
            msg_a4 = Float64()
            msg_a5 = Float64()
            msg_a6 = Float64()

###### Robot arm full position (a1= 0, a2=-90, a3= 0, a4= 0, a5= 0 , a6= 0) ######
###### Robot initial position (a1= 0, a2=-95, a3= 50, a4= 0, a5= 100 , a6= 0) ######

###### Robot joint limits in rads (a1= , a2= , a3= , a4= ,a5= 2.09, a6=  ) ######

            msg_a1.data = self.ja1
            msg_a2.data = self.ja2
            msg_a3.data = self.ja3
            msg_a4.data = self.ja4
            msg_a5.data = self.ja5
            msg_a6.data = self.ja6

            self.pub_a1.publish(msg_a1)
            self.pub_a2.publish(msg_a2)
            self.pub_a3.publish(msg_a3)
            self.pub_a4.publish(msg_a4)
            self.pub_a5.publish(msg_a5)
            self.pub_a6.publish(msg_a6)

if __name__ == "__main__" :
    node = Position()
    node.spin()