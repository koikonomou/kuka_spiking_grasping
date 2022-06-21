import math
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates
from std_msgs.msg import Float64
from table_exploration.msg import Collision


class Collision(object):
	def __init__(self):
		rospy.init_node('collision_detection', anonymous = True)
		self.rate = rospy.get_param("rate",10)
		self.joint_topic = rospy.get_param("joint_topic", "/gazebo/link_states")
		self.joint_sub = rospy.Subscriber(self.joint_topic, LinkStates, self.callback, queue_size=10)
		self.pub = rospy.Publisher('/collision_detection', Collision, queue_size=1)
		self.goal_pos = [0.5,0.0,1.0]

	def dot(self, v, w):
	    x,y,z = v
	    X,Y,Z = w
	    return x*X + y*Y + z*Z

	def length(self, v):
	    x,y,z = v
	    return math.sqrt(x*x + y*y + z*z)

	def vector(self, b, e):
	    x,y,z = b
	    X,Y,Z = e
	    return (X-x, Y-y, Z-z)

	def unit(self, v):
	    x,y,z = v
	    mag = self.length(v)
	    return (x/mag, y/mag, z/mag)

	def distance(self, p0, p1):
	    return self.length(self.vector(p0,p1))

	def scale(self, v, sc):
	    x,y,z = v
	    return (x * sc, y * sc, z * sc)

	def add(self, v, w):
		x,y,z = v
		X,Y,Z = w
		return (x+X, y+Y, z+Z)

	def pnt2line(self, pnt, start, end):
		line_vec = self.vector(start, end)
		pnt_vec = self.vector(start, pnt)
		line_len = self.length(line_vec)
		line_unitvec = self.unit(line_vec)
		pnt_vec_scaled = self.scale(pnt_vec, 1.0/line_len)
		t = self.dot(line_unitvec, pnt_vec_scaled)    
		if t < 0.0:
		    t = 0.0
		elif t > 1.0:
		    t = 1.0
		nearest = self.scale(line_vec, t)
		dist = self.distance(nearest, pnt_vec)
		# nearest = add(nearest, start)
		return dist

	def callback(self, msg):
        # rate = rospy.Rate(self.rate)

		self.pos_j0 = [msg.pose[1].position.x, msg.pose[1].position.y, msg.pose[1].position.z]
		self.pos_j1 = [msg.pose[2].position.x, msg.pose[2].position.y, msg.pose[2].position.z]
		self.pos_j2 = [msg.pose[3].position.x, msg.pose[3].position.y, msg.pose[3].position.z]
		self.pos_j3 = [msg.pose[4].position.x, msg.pose[4].position.y, msg.pose[4].position.z]
		self.pos_j4 = [msg.pose[5].position.x, msg.pose[5].position.y, msg.pose[5].position.z]
		self.pos_j5 = [msg.pose[6].position.x, msg.pose[6].position.y, msg.pose[6].position.z]

	def spin(self):
		rate = rospy.Rate(self.rate)

		while not rospy.is_shutdown():
			rate.sleep()

			if self.pos_j0 and self.pos_j1 and self.pos_j3 and self.pos_j4 and self.pos_j5 is None:
				rospy.logwarn_throttle(2, "no image")
				continue
			try:
				# height of the end effector have to be always >0.85 inorder to avoid table collision
				self.height_endef = self.pos_j5[2]
				
				# calculate joint 3 from link01,link12
				self.distj3_link1 = self.pnt2line(self.pos_j3, self.pos_j0, self.pos_j1)
				self.distj3_link2 = self.pnt2line(self.pos_j3, self.pos_j1, self.pos_j2)
				
				# calculate distance from joint 2 to link 1
				self.distj2_link1 = self.pnt2line(self.pos_j2, self.pos_j0, self.pos_j1) 
				p1 = np.array(self.pos_j5)
				p2 = np.array(self.goal_pos)
				squared_dist = np.sum((p1-p2)**2, axis=0)
				actual_dist = np.sqrt(squared_dist)
				# calculate distance from the end effector to the base
				self.distj5_link1 = self.pnt2line(self.pos_j5,self.pos_j0,self.pos_j1)

				""" if end effector is lower than table then it has collide 
				and the distance will be -100"""
				if self.height_endef < 0.85:
					msg = Collision()
					msg.goal_dist = actual_dist
					msg.have_collide = -5
					self.pub.publish(msg)
				elif self.distj5_link1 < 0.20:
					msg = Collision()
					msg.goal_dist = actual_dist
					msg.have_collide = -5
					self.pub.publish(msg)
				else:
					msg = Collision()
					msg.goal_dist = actual_dist
					msg.have_collide = 0
					self.pub.publish(msg)
			except ValueError:
				rospy.logwarn_throttle(2, "object detection error")



if __name__ == "__main__":
    node = Collision()
    node.spin()