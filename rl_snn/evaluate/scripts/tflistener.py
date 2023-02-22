import rospy
import math
import tf



if __name__ == '__main__':
    rospy.init_node('look_up_kuka_tf')

    listener_a = tf.TransformListener()
    listener_b = tf.TransformListener()
    listener_c = tf.TransformListener()
    trans_a = []
    trans_b = []
    trans_c = []
    rot_a = []
    rot_b = []
    rot_c=[]
    rate = rospy.Rate(10.0)
    while not trans_a:
        try:
            (trans_a,rot_a) = listener_a.lookupTransform('/camera_link', '/world', rospy.Time(0))
        except:
            continue
    while not trans_b:
        try:
            (trans_b,rot_b) = listener_a.lookupTransform('/box_link', '/world', rospy.Time(0))
        except:
            continue
    while not trans_c:
        try:
            (trans_c,rot_c) = listener_a.lookupTransform('/camera_link', '/box_link', rospy.Time(0))
        except:
            continue
        rate.sleep()
    print 'Translation camera-world: ' , trans_a
    print 'Rotation camera-world: ' , rot_a
    print 'Translation box-world: ' , trans_b
    print 'Rotation box-world: ' , rot_b
    print 'Translation camera-box: ' , trans_c
    print 'Rotation camera-box: ' , rot_c