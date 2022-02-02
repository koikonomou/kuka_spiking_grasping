import math
import rospy
from std_msgs.msg import Float64


pose_publisher = None


def pose_publisher():
    pub_a1 = rospy.Publisher('/kuka_kr4r600/joint_a1_position_controller/command', Float64, queue_size=10)
    pub_a2 = rospy.Publisher('/kuka_kr4r600/joint_a2_position_controller/command', Float64, queue_size=10)
    pub_a3 = rospy.Publisher('/kuka_kr4r600/joint_a3_position_controller/command', Float64, queue_size=10)
    pub_a4 = rospy.Publisher('/kuka_kr4r600/joint_a4_position_controller/command', Float64, queue_size=10)
    pub_a5 = rospy.Publisher('/kuka_kr4r600/joint_a5_position_controller/command', Float64, queue_size=10)
    pub_a6 = rospy.Publisher('/kuka_kr4r600/joint_a6_position_controller/command', Float64, queue_size=10)

    rospy.init_node('pose_publisher', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        msg_a1 = Float64()
        msg_a2 = Float64()
        msg_a3 = Float64()
        msg_a4 = Float64()
        msg_a5 = Float64()
        msg_a6 = Float64()

        msg_a1.data = -1
        msg_a2.data = -1
        msg_a3.data = 1.2
        msg_a4.data = -0.2
        msg_a5.data = 1
        msg_a6.data = 0.0


        pub_a1.publish(msg_a1)
        pub_a2.publish(msg_a2)
        pub_a3.publish(msg_a3)
        pub_a4.publish(msg_a4)
        pub_a5.publish(msg_a5)
        pub_a6.publish(msg_a6)
        rate.sleep()

if __name__ == '__main__':
    try:
        pose_publisher()
    except rospy.ROSInterruptException:
        pass