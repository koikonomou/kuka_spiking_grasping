# ! /usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64

pub = None

def callback(msg):
    dist = min(msg.ranges[288:431])
    distance(dist)
def distance(dist):
    pub_msg = Float64()
    pub_msg.data = dist
    pub.publish(pub_msg)

def main():
    global pub
    rospy.init_node('collision')
    pub = rospy.Publisher('/collision', Float64, queue_size=1)
    sub = rospy.Subscriber('/kuka/laser/scan', LaserScan, callback)
    rospy.spin()


if __name__=='__main__':
    main()