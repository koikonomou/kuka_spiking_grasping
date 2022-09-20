# ! /usr/bin/env python3
import math
import rospy
import numpy as np
from gazebo_msgs.msg import LinkStates
from std_msgs.msg import Float64, Int32
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

class MarkerArrayVis(object):
	def __init__(self):
		rospy.init_node('visualization_markers', anonymous = True)
		self.rate = rospy.get_param("rate",10)
		self.callback_init = False
		# self.color_cb_init = False
		self.change_color = rospy.Subscriber("/change_marker_color", Int32, self.color_cb, queue_size=10)
		self.gazebo_joint_topic = rospy.get_param("gazebo_joint_topic", "/gazebo/link_states")
		self.joint_sub = rospy.Subscriber(self.gazebo_joint_topic, LinkStates, self.callback, queue_size=10)
		self.pub = rospy.Publisher("/markers", MarkerArray, queue_size = 2)
		self.target_pose = [0.5, 0.0, 0.85]
		self.jointa2_color = [1.0,0.0,0.0,1.0]
		self.jointa3_color = [1.0,0.0,0.0,1.0]
		while not self.callback_init:
			continue
		rospy.loginfo("Finish Subscriber Init ...")

	def color_cb(self,msg):
		# if self.color_cb_init is False:
		# 	self.color_cb_init = True
		if msg.data == 0:
			self.jointa2_color = [1.0,0.0,0.0,1.0]
			self.jointa3_color = [1.0,0.0,0.0,1.0]
		elif msg.data == 1:
			self.jointa2_color = [0.0,1.0,0.0,1.0]
			self.jointa3_color = [1.0,0.0,0.0,1.0]
		elif msg.data == 2:
			self.jointa2_color = [0.0,1.0,0.0,1.0]
			self.jointa3_color = [0.0,1.0,0.0,1.0]

	def callback(self,msg):
		if self.callback_init is False:
			self.callback_init = True
		self.pos_endeffector = [msg.pose[7].position.x, msg.pose[7].position.y, msg.pose[7].position.z]

	def spin(self):
		rate =  rospy.Rate(self.rate)

		while not rospy.is_shutdown():
			rate.sleep()

			# Calculate equidistance points between end effector and target point.
			point_ax = self.pos_endeffector[0]
			point_ay = self.pos_endeffector[1]
			point_az = self.pos_endeffector[2]

			point_dx = self.target_pose[0]
			point_dy = self.target_pose[1]
			point_dz = self.target_pose[2]

			point_bx = point_ax+(point_dx-point_ax)/3
			point_by = point_ay+(point_dy-point_ay)/3
			point_bz = point_az+(point_dz-point_az)/3
			point_b = [point_bx, point_by, point_bz]

			point_cx = point_ax+2*(point_dx-point_ax)/3
			point_cy = point_ay+2*(point_dy-point_ay)/3
			point_cz = point_az+2*(point_dz-point_az)/3
			point_c = [point_cx, point_cy, point_cz]


			markerar = MarkerArray()

			marker_a = Marker()
			marker_a.header.frame_id = "world"
			marker_a.header.stamp = rospy.Time.now()

			# Shape (mesh resource type - 10)
			marker_a.type = 2
			marker_a.id = 0
			marker_a.action = 0


			# Scale
			marker_a.scale.x = 0.07
			marker_a.scale.y = 0.07
			marker_a.scale.z = 0.07

			# Color
			marker_a.color.r = 0.0
			marker_a.color.g = 1.0
			marker_a.color.b = 0.0
			marker_a.color.a = 1.0

			# Pose
			marker_a.pose.position.x = point_ax
			marker_a.pose.position.y = point_ay
			marker_a.pose.position.z = point_az
			marker_a.pose.orientation.x = 0.0
			marker_a.pose.orientation.y = 0.0
			marker_a.pose.orientation.z = 0.0
			marker_a.pose.orientation.w = 1.0
			markerar.markers.append(marker_a)

			marker_b = Marker()
			marker_b.header.frame_id = "world"
			marker_b.header.stamp = rospy.Time.now()

			# Shape (mesh resource type - 10)
			marker_b.type = 2
			marker_b.id = 1
			marker_b.action = 0


			# Scale
			marker_b.scale.x = 0.07
			marker_b.scale.y = 0.07
			marker_b.scale.z = 0.07

			# Color
			marker_b.color.r = self.jointa2_color[0]
			marker_b.color.g = self.jointa2_color[1]
			marker_b.color.b = self.jointa2_color[2]
			marker_b.color.a = self.jointa2_color[3]

			# Pose
			marker_b.pose.position.x = point_bx
			marker_b.pose.position.y = point_by
			marker_b.pose.position.z = point_bz
			marker_b.pose.orientation.x = 0.0
			marker_b.pose.orientation.y = 0.0
			marker_b.pose.orientation.z = 0.0
			marker_b.pose.orientation.w = 1.0
			markerar.markers.append(marker_b)

			marker_c = Marker()
			marker_c.header.frame_id = "world"
			marker_c.header.stamp = rospy.Time.now()

			# Shape (mesh resource type - 10)
			marker_c.type = 2
			marker_c.id = 2
			marker_c.action = 0


			# Scale
			marker_c.scale.x = 0.07
			marker_c.scale.y = 0.07
			marker_c.scale.z = 0.07

			# Color
			marker_c.color.r = self.jointa3_color[0]
			marker_c.color.g = self.jointa3_color[1]
			marker_c.color.b = self.jointa3_color[2]
			marker_c.color.a = self.jointa3_color[3]

			# Pose
			marker_c.pose.position.x = point_cx
			marker_c.pose.position.y = point_cy
			marker_c.pose.position.z = point_cz
			marker_c.pose.orientation.x = 0.0
			marker_c.pose.orientation.y = 0.0
			marker_c.pose.orientation.z = 0.0
			marker_c.pose.orientation.w = 1.0
			markerar.markers.append(marker_c)

			marker_d = Marker()
			marker_d.header.frame_id = "world"
			marker_d.header.stamp = rospy.Time.now()

			# Shape (mesh resource type - 10)
			marker_d.type = 2
			marker_d.id = 3
			marker_d.action = 0


			# Scale
			marker_d.scale.x = 0.07
			marker_d.scale.y = 0.07
			marker_d.scale.z = 0.07

			# Color
			marker_d.color.r = 0.0
			marker_d.color.g = 1.0
			marker_d.color.b = 0.0
			marker_d.color.a = 1.0

			# Pose
			marker_d.pose.position.x = point_dx
			marker_d.pose.position.y = point_dy
			marker_d.pose.position.z = point_dz
			marker_d.pose.orientation.x = 0.0
			marker_d.pose.orientation.y = 0.0
			marker_d.pose.orientation.z = 0.0
			marker_d.pose.orientation.w = 1.0
			markerar.markers.append(marker_d)

			self.pub.publish(markerar)

			# except ValueError:
			# 	rospy.logwarn_throttle(2, "Marker visualization error")



if __name__ == "__main__":
    node = MarkerArrayVis()
    node.spin()