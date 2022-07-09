import rospy
import math
import copy
import random
import numpy as np
from std_srvs.srv import Empty
from std_msgs.msg import Float64
from shapely.geometry import Point
from table_exploration.msg import Distance
from table_exploration.msg import Collision

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.msg import LinkStates, LinkState , ModelState

class GazeboEnvironment:

    """
    Class for Gazebo Environment

    Main Function:
        1. Reset: Rest environment at the end of each episode
        and generate new goal position for next episode

        2. Step: Execute new action and return state
     """
    def __init__(self,
                 # distance_num = 1,
                 goal_dis_min=0.1,
                 obs_near_th=0.15,
                 goal_near_th=0.15,
                 goal_reward=10,
                 col_reward=-20,
                 goal_dis_amp=5,
                 step_time=0.1):
        """
        :param goal_dis_min: minimal distance of goal distance
        :param goal_dis_scale: goal distance scale
        :param obs_near_th: Threshold for near an obstacle
        :param goal_near_th: Threshold for near an goal
        :param goal_reward: reward for reaching goal
        :param col_reward: reward for reaching obstacle
        :param goal_dis_amp: amplifier for goal distance change
        :param step_time: time for a single step (DEFAULT: 0.1 seconds)
        """

        self.goal_pos_list = None
        self.obstacle_poly_list = None
        self.robot_init_pose_list = None
        # self.distance_num = distance_num
        self.goal_dis_min = goal_dis_min
        self.obs_near_th = obs_near_th
        self.goal_near_th = goal_near_th
        self.goal_reward = goal_reward
        self.col_reward = col_reward
        self.goal_dis_amp = goal_dis_amp
        self.step_time = step_time
        # Robot State (6 joints)
        # 5 (joints) * [ 3  (position values = x,y,z) + 4 (orientation values = x,y,z,w) ] = 35 values
        self.robot_pose = [0.]*35
        # 5 (joints) * [3 (linear x,y,z) + 3 (angular x,y,z)] = 30 values
        self.robot_speed = [0.]*30
        self.robot_goal_dist = 0
        # self.found_obj = -100 # IF 0 object dont is the scene view elif 1 object in scene view.
        self.collision = 0 # Laserscan data for collision. (Distance)
        self.scan_dist = 0
        self.have_collide = 0
        self.robot_state_init = False
        self.robot_distance_init = False
        self.collision_dist_init = False
        self.collision_init = False
        # Change this from goal position to goal state
        self.goal_dis_pre = 0  # Last step goal distance
        self.goal_dis_cur = 0  # Current step goal 
        self.scan_dist_pre = 0
        self.scan_dist_cur = 0
        # self.init_pose_list = [0.0,0.0,0.0,0.0,0.0]
        # Subscriber
        # rospy.Subscriber('gazebo/model_states', ModelStates, self._robot_state_cb)
        rospy.Subscriber('gazebo/link_states', LinkStates, self._robot_link_cb )
        rospy.Subscriber('/kuka/collision', Float64, self.collision_dist_cb)
        rospy.Subscriber('/kuka/box/distance', Distance, self._robot_distance_cb )
        rospy.Subscriber('/collision_detection', Collision, self.collision_cb)

        """ ROBOT LINK NAMES: ['ground_plane::link', 'kuka_kr4r600::table_top_link',
         'kuka_kr4r600::link_1', 'kuka_kr4r600::link_2','kuka_kr4r600::link_3', 
         'kuka_kr4r600::link_4', 'kuka_kr4r600::link_5', 'kuka_kr4r600::link_6'] """

        # We don't add joint a_6 because it has the camera.
        # TODO CONSIDER TO ADD THE 6th joint

        # Publisher
        self.pub_a1 = rospy.Publisher('/kuka_kr4r600/joint_a1_position_controller/command', Float64, queue_size=1)
        self.pub_a2 = rospy.Publisher('/kuka_kr4r600/joint_a2_position_controller/command', Float64, queue_size=1)
        self.pub_a3 = rospy.Publisher('/kuka_kr4r600/joint_a3_position_controller/command', Float64, queue_size=1)
        self.pub_a4 = rospy.Publisher('/kuka_kr4r600/joint_a4_position_controller/command', Float64, queue_size=1)
        self.pub_a5 = rospy.Publisher('/kuka_kr4r600/joint_a5_position_controller/command', Float64, queue_size=1)
        # self.pub_a6 = rospy.Publisher('/kuka_kr4r600/joint_a6_position_controller/command', Float64, queue_size=5)

        # Service
        self.pause_gazebo = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.unpause_gazebo = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        # Init Subscriber
        while not self.robot_state_init:
            continue
        while not self.robot_distance_init:
            continue
        while not self.collision_dist_init:
            continue
        while not self.collision_init:
            continue
        rospy.loginfo("Finish Subscriber Init...")

    def step(self, action):
        """
        Step Function for the Environment

        Take a action for the robot and return the updated state
        :param action: action taken
        :return: state, reward, done
        """
        rospy.wait_for_service('gazebo/unpause_physics')
        try:
            self.unpause_gazebo()
        except rospy.ServiceException as e:
            print("Unpause Service Failed: %s" % e)
        '''
        First give action to robot and let robot execute and get next state
        '''
        move_joint_a1 = Float64()
        move_joint_a1.data = action[0]
        move_joint_a2 = Float64()
        move_joint_a2.data = action[1]
        move_joint_a3 = Float64()
        move_joint_a3.data = action[2]
        move_joint_a4 = Float64()
        move_joint_a4.data = action[3]
        move_joint_a5 = Float64()
        move_joint_a5.data = action[4]
        # move_joint_a6 = Float64
        # move_joint_a6.data = action[5]


        self.pub_a1.publish(move_joint_a1)
        self.pub_a2.publish(move_joint_a2)
        self.pub_a3.publish(move_joint_a3)
        self.pub_a4.publish(move_joint_a4)
        self.pub_a5.publish(move_joint_a5)
        # self.pub_a6.publish(move_joint_a6)
        rospy.sleep(self.step_time)


        next_robot_state = self._get_next_robot_state()
        rospy.wait_for_service('gazebo/pause_physics')
        try:
            self.pause_gazebo()
        except rospy.ServiceException as e:
            print("Pause Service Failed: %s" % e)
        '''
        Then stop the simulation
        1. Transform Robot State to snn State
        2. Compute Reward of the action
        3. Compute if the episode is ended
        '''
        self.goal_dis_cur = self.actual_dist
        self.scan_dist_cur = self.scan_dist
        have_collide_now = self.have_collide
        state = self._robot_state_2_snn_state(next_robot_state)
        reward, done = self._compute_reward(state, have_collide_now)
        self.goal_dis_pre = self.goal_dis_cur
        self.scan_dist_pre = self.scan_dist_cur
        return state, reward, done

    def reset(self):

        assert self.robot_init_pose_list is not None
        # assert ita < len(self.goal_pos_list)
        rospy.wait_for_service('gazebo/unpause_physics')
        try:
            self.unpause_gazebo()
        except rospy.ServiceException as e:
            print("Unpause Service Failed: %s" % e)
        self.pub_a1.publish(Float64())
        self.pub_a2.publish(Float64())
        self.pub_a3.publish(Float64())
        self.pub_a4.publish(Float64())
        self.pub_a5.publish(Float64())

        self.pub_a1.publish(self.robot_init_pose_list[0])
        self.pub_a2.publish(self.robot_init_pose_list[1])
        self.pub_a3.publish(self.robot_init_pose_list[2])
        self.pub_a4.publish(self.robot_init_pose_list[3])
        self.pub_a5.publish(self.robot_init_pose_list[4])
        """ ROBOT LINK NAMES: ['ground_plane::link', 'kuka_kr4r600::table_top_link',
         'kuka_kr4r600::link_1', 'kuka_kr4r600::link_2','kuka_kr4r600::link_3', 
         'kuka_kr4r600::link_4', 'kuka_kr4r600::link_5', 'kuka_kr4r600::link_6'] """
        rospy.sleep(0.5)
        '''
        Transfer the initial robot state to the state for the Agent
        '''
        rob_state = self._get_next_robot_state()
        rospy.wait_for_service('gazebo/pause_physics')
        try:
            self.pause_gazebo()
        except rospy.ServiceException as e:
            print("Pause Service Failed: %s" % e)
        # goal_dis = self._compute_dis_2_goal(rob_state[0])
        self.goal_dis_pre = self.actual_dist
        self.goal_dis_cur = self.actual_dist
        self.scan_dist_pre = self.scan_dist
        self.scan_dist_cur = self.scan_dist
        state = self._robot_state_2_snn_state(rob_state)
        return state

    def reset_environment(self, init_poslist):
        self.robot_init_pose_list = init_poslist
        rospy.wait_for_service('gazebo/unpause_physics')
        try:
            self.unpause_gazebo()
        except rospy.ServiceException as e:
            print("Unpause Service Failed: %s" % e)
        self.pub_a1.publish(self.robot_init_pose_list[0])
        self.pub_a2.publish(self.robot_init_pose_list[1])
        self.pub_a3.publish(self.robot_init_pose_list[2])
        self.pub_a4.publish(self.robot_init_pose_list[3])
        self.pub_a5.publish(self.robot_init_pose_list[4])
        rospy.wait_for_service('gazebo/pause_physics')
        try:
            self.pause_gazebo()
        except rospy.ServiceException as e:
            print("Pause Service Failed: %s" % e)

    def _get_next_robot_state(self):
        """
        Get the combination of state after execute the action for a certain time
        State will be: [robot_pose, robot_spd, col_dist]
        :return: state
        """
        tmp_robot_pose = copy.deepcopy(self.robot_pose)
        tmp_robot_speed = copy.deepcopy(self.robot_speed)
        tmp_goal_dist = self.scan_dist
        # tmp_goal_dist = self.found_obj
        state = [tmp_robot_pose, tmp_robot_speed, tmp_goal_dist]
        return state

    def _robot_state_2_snn_state(self, state):
        # State 0 include the pose(position and orientation) of all joints
        # State 1 include the speed(linear and angular) of all joints
        # State 2 include scan distance
        snn_state = [[self.goal_dis_cur], state[0], state[1], [state[2]]]
        return snn_state

    def _compute_reward(self, state, have_collide_now):
        done = False

        near_obstacle = False
        goal = False
        # First check distance from scanner if scan show that it is near obstacle
        if have_collide_now != 0:
            near_obstacle = True
        elif self.scan_dist < 0.2 and self.found_obj == 0:
            near_obstacle = True
        elif self.actual_dist < 0.05:
            near_obstacle = True
        # elif have_collide_now < 0.2 and self.found_obj == 1:
        #     print("Reached goal")
        #     goal = True
            
        '''
        Assign Rewards
        '''

        if self.goal_dis_cur < 0.1:
            reward = self.goal_reward
            done = True
        elif near_obstacle:
            reward = self.col_reward
            done = True
        else:
            reward = (self.goal_dis_amp + (self.found_obj * self.goal_dis_amp))/self.scan_dist * 1/(self.goal_dis_cur)
        return reward, done

    def _robot_link_cb(self,msg):

        """
        Callback function for robot link state
        :param msg: message
        """

        if self.robot_state_init is False:
            self.robot_state_init = True
            # Pose[0] belongs to ground plane and pose[1] to table_link
            # P_j robot state has position and orientation for each joint
            pos_j1 = [msg.pose[2].position.x, msg.pose[2].position.y, msg.pose[2].position.z]
            pos_j2 = [msg.pose[3].position.x, msg.pose[3].position.y, msg.pose[3].position.z]
            pos_j3 = [msg.pose[4].position.x, msg.pose[4].position.y, msg.pose[4].position.z]
            pos_j4 = [msg.pose[5].position.x, msg.pose[5].position.y, msg.pose[5].position.z]
            pos_j5 = [msg.pose[6].position.x, msg.pose[6].position.y, msg.pose[6].position.z]

            quat_j1 = [msg.pose[2].orientation.x, msg.pose[2].orientation.y, msg.pose[2].orientation.z, msg.pose[2].orientation.w]
            quat_j2 = [msg.pose[3].orientation.x, msg.pose[3].orientation.y, msg.pose[3].orientation.z, msg.pose[3].orientation.w]
            quat_j3 = [msg.pose[4].orientation.x, msg.pose[4].orientation.y, msg.pose[4].orientation.z, msg.pose[4].orientation.w]
            quat_j4 = [msg.pose[5].orientation.x, msg.pose[5].orientation.y, msg.pose[5].orientation.z, msg.pose[5].orientation.w]
            quat_j5 = [msg.pose[6].orientation.x, msg.pose[6].orientation.y, msg.pose[6].orientation.z, msg.pose[6].orientation.w]

            # V_j has linear and angular velocities for each joint
            linear_j1 = [msg.twist[2].linear.x, msg.twist[2].linear.y, msg.twist[2].linear.z]
            linear_j2 = [msg.twist[3].linear.x, msg.twist[3].linear.y, msg.twist[3].linear.z]
            linear_j3 = [msg.twist[4].linear.x, msg.twist[4].linear.y, msg.twist[4].linear.z]
            linear_j4 = [msg.twist[5].linear.x, msg.twist[5].linear.y, msg.twist[5].linear.z]
            linear_j5 = [msg.twist[6].linear.x, msg.twist[6].linear.y, msg.twist[6].linear.z]

            angular_j1 = [msg.twist[2].angular.x, msg.twist[2].angular.y, msg.twist[2].angular.z]
            angular_j2 = [msg.twist[3].angular.x, msg.twist[3].angular.y, msg.twist[3].angular.z]
            angular_j3 = [msg.twist[4].angular.x, msg.twist[4].angular.y, msg.twist[4].angular.z]
            angular_j4 = [msg.twist[5].angular.x, msg.twist[5].angular.y, msg.twist[5].angular.z]
            angular_j5 = [msg.twist[6].angular.x, msg.twist[6].angular.y, msg.twist[6].angular.z]

            self.robot_pose = pos_j1+quat_j1+pos_j2+quat_j2+pos_j3+quat_j3+pos_j4+quat_j4+pos_j5+quat_j5
            self.robot_speed = linear_j1+angular_j1+linear_j2+angular_j2+linear_j3+angular_j3+linear_j4+angular_j4+linear_j5+angular_j5


    def _robot_distance_cb(self,msg):
        if self.robot_distance_init is False:
            self.robot_distance_init = True
        # self.robot_goal_dist = msg.data
        self.found_obj = msg.havered

    def collision_dist_cb(self,msg):
        if self.collision_dist_init is False:
            self.collision_dist_init = True
        self.scan_dist = msg.data

    def collision_cb(self,msg):
        if self.collision_init is False:
            self.collision_init = True
        self.have_collide = msg.have_collide
        self.actual_dist = msg.goal_dist