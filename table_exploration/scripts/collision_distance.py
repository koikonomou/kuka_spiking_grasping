# ! /usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64


def callback(msg):
    dist = min(msg.ranges[288:431])
    test= Float64()
    test.data = dist
    # print(dist)
    pub.publish(test)

def main():
    global pub
    rospy.init_node('collision')
    pub = rospy.Publisher('/kuka/collision', Float64, queue_size=1)
    sub = rospy.Subscriber('/kuka/laser/scan', LaserScan, callback)
    rospy.spin()


if __name__=='__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass