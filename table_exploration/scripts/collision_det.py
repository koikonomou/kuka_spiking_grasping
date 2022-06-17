import rospy
import numpy as np
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates

class Collision(object):
	def __init__(self):
		rospy.init_node('collision_detection', anonymous = True)
		self.rate = rospy.get_param("rate",10)
		self.joint_topic = rospy.get_param("joint_topic", "/joint_states")
		self.joint_sub = rospy.Subscriber(self.joint_topic, LinkStates, self.callback queue_size=10)
		# self.col_pb = rospy.get_param("col_pb",)
		# self.pub = rospy.Publisher(self.col_pb)

	def callback(self, msg):
		self.pos_j0 = [msg.pose[1].position.x, msg.pose[1].position.y, msg.pose[1].position.z]
		self.pos_j1 = [msg.pose[2].position.x, msg.pose[2].position.y, msg.pose[2].position.z]
		self.pos_j2 = [msg.pose[3].position.x, msg.pose[3].position.y, msg.pose[3].position.z]
		self.pos_j3 = [msg.pose[4].position.x, msg.pose[4].position.y, msg.pose[4].position.z]
		self.pos_j4 = [msg.pose[5].position.x, msg.pose[5].position.y, msg.pose[5].position.z]
		self.pos_j5 = [msg.pose[6].position.x, msg.pose[6].position.y, msg.pose[6].position.z]

		self.link01 = self.pos_j0[:-1] - self.pos_j1[:-1]
		self.dist01 = np.divide(self.link01,(np.hypot(self.link01[:,0], self.link01[:,1]).reshape(-1,1)))
