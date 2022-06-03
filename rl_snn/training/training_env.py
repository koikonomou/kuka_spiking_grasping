import rospy
import math
import copy
import random
import numpy as np
from std_srvs.srv import Empty
from std_msgs.msg import Float64
from shapely.geometry import Point
from table_exploration.msg import Distance
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
                 col_reward=-5,
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
        self.found_obj = -100 # IF 0 object dont is the scene view elif 1 object in scene view.
        self.collision = 0 # Laserscan data for collision. (Distance)
        self.scan_dist = 0
        self.robot_state_init = False
        self.robot_distance_init = False
        self.collision_dist_init = False
        # Goal Position
        self.goal_position = [0.5, 0., 0.9]
        # Change this from goal position to goal state
        self.goal_state = [1 , 0.1] # The goal state is [havered,distance_from_object]. (havered means that redobject havefound)
        self.goal_dis_pre = 0  # Last step goal distance
        self.goal_dis_cur = 0  # Current step goal 
        self.scan_dist_pre = 0
        self.scan_dist_cur = 0
        # Subscriber
        # rospy.Subscriber('gazebo/model_states', ModelStates, self._robot_state_cb)
        rospy.Subscriber('gazebo/link_states', LinkStates, self._robot_link_cb )
        rospy.Subscriber('/kuka/collision', Float64, self.collision_dist_cb)
        rospy.Subscriber('/kuka/box/distance', Distance, self._robot_distance_cb )

        """ ROBOT LINK NAMES: ['ground_plane::link', 'kuka_kr4r600::table_top_link',
         'kuka_kr4r600::link_1', 'kuka_kr4r600::link_2','kuka_kr4r600::link_3', 
         'kuka_kr4r600::link_4', 'kuka_kr4r600::link_5', 'kuka_kr4r600::link_6'] """

        # We don't add joint a_6 because it has the camera.
        # TODO CONSIDER TO ADD THE 6th joint

        # Publisher
        self.pub_a1 = rospy.Publisher('/kuka_kr4r600/joint_a1_position_controller/command', Float64, queue_size=5)
        self.pub_a2 = rospy.Publisher('/kuka_kr4r600/joint_a2_position_controller/command', Float64, queue_size=5)
        self.pub_a3 = rospy.Publisher('/kuka_kr4r600/joint_a3_position_controller/command', Float64, queue_size=5)
        self.pub_a4 = rospy.Publisher('/kuka_kr4r600/joint_a4_position_controller/command', Float64, queue_size=5)
        self.pub_a5 = rospy.Publisher('/kuka_kr4r600/joint_a5_position_controller/command', Float64, queue_size=5)
        # self.pub_a6 = rospy.Publisher('/kuka_kr4r600/joint_a6_position_controller/command', Float64, queue_size=5)

        # Service
        self.pause_gazebo = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.unpause_gazebo = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.set_model_target = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        self.set_link_target = rospy.ServiceProxy('gazebo/set_link_state', SetLinkState)
        self.reset_simulation = rospy.ServiceProxy('gazebo/reset_simulation', Empty)

        # Init Subscriber
        while not self.robot_state_init:
            continue
        while not self.robot_distance_init:
            continue
        while not self.collision_dist_init:
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

        rospy.sleep(self.step_time)

        self.pub_a1.publish(move_joint_a1)
        self.pub_a2.publish(move_joint_a2)
        self.pub_a3.publish(move_joint_a3)
        self.pub_a4.publish(move_joint_a4)
        self.pub_a5.publish(move_joint_a5)
        # self.pub_a6.publish(move_joint_a6)


        next_robot_state = self._get_next_robot_state()
        rospy.wait_for_service('gazebo/pause_physics')
        try:
            self.pause_gazebo()
        except rospy.ServiceException as e:
            print("Pause Service Failed: %s" % e)
        '''
        Then stop the simulation
        1. Transform Robot State to DDPG State
        2. Compute Reward of the action
        3. Compute if the episode is ended
        '''
        goal_dis = self.robot_goal_dist
        scan_dist = self.scan_dist
        self.goal_dis_cur = goal_dis
        self.scan_distance_cur = scan_dist
        state = self._robot_state_2_ddpg_state(next_robot_state)
        reward, done = self._compute_reward(state)
        self.goal_dis_pre = self.goal_dis_cur
        self.scan_dist_pre = self.scan_dist_cur
        return state, reward, done

    def reset(self, ita):

        """
        Reset Function to reset simulation at start of each episode

        Return the initial state after reset
        :param ita: number of route to reset to
        :return: state
        """

        assert self.robot_init_pose_list is not None
        # assert ita < len(self.goal_pos_list)
        rospy.wait_for_service('gazebo/unpause_physics')
        try:
            self.unpause_gazebo()
        except rospy.ServiceException as e:
            print("Unpause Service Failed: %s" % e)
        '''
        First choose new goal position and set target model to goal
        '''
        # self.goal_position = self.goal_pos_list[ita]
        # target_msg = ModelState()
        # target_msg.model_name = 'target'
        # target_msg.pose.position.x = self.goal_position[0]
        # target_msg.pose.position.y = self.goal_position[1]
        # target_msg.pose.position.z = self.goal_position[2]
        # rospy.wait_for_service('gazebo/set_model_state')
        # try:
        #     resp = self.set_model_target(target_msg)
        # except rospy.ServiceException as e:
        #     print("Set Target Service Failed: %s" % e)
        '''
        Then reset robot state and get initial state
        '''
        # self.pub_action.publish(Twist())
        self.pub_a1.publish(Float64())
        self.pub_a2.publish(Float64())
        self.pub_a3.publish(Float64())
        self.pub_a4.publish(Float64())
        self.pub_a5.publish(Float64())
        """ ROBOT LINK NAMES: ['ground_plane::link', 'kuka_kr4r600::table_top_link',
         'kuka_kr4r600::link_1', 'kuka_kr4r600::link_2','kuka_kr4r600::link_3', 
         'kuka_kr4r600::link_4', 'kuka_kr4r600::link_5', 'kuka_kr4r600::link_6'] """

        robot_init_pose = self.robot_init_pose_list
        # robot_init_quat = self._euler_2_quat(yaw=robot_init_pose[2])

        link1_msg = LinkState()
        link1_msg.link_name = 'kuka_kr4r600::link_1'
        link1_msg.pose.position.x = robot_init_pose[0]
        link1_msg.pose.position.y = robot_init_pose[1]
        link1_msg.pose.position.z = robot_init_pose[2]
        link1_msg.pose.orientation.x = robot_init_pose[3]
        link1_msg.pose.orientation.y = robot_init_pose[4]
        link1_msg.pose.orientation.z = robot_init_pose[5]
        link1_msg.pose.orientation.w = robot_init_pose[6]
        link1_msg.reference_frame = 'kuka_kr4r600::base_link'
        try:
            resp = self.set_link_target(link1_msg)
        except rospy.ServiceException as e:
            print("Set Link1 Target Service Failed: %s" % e)

        link2_msg = LinkState()
        link2_msg.link_name = 'kuka_kr4r600::link_2'
        link2_msg.pose.position.x = robot_init_pose[7]
        link2_msg.pose.position.y = robot_init_pose[8]
        link2_msg.pose.position.z = robot_init_pose[9]
        link2_msg.pose.orientation.x = robot_init_pose[10]
        link2_msg.pose.orientation.y = robot_init_pose[11]
        link2_msg.pose.orientation.z = robot_init_pose[12]
        link2_msg.pose.orientation.w = robot_init_pose[12]
        link2_msg.reference_frame = 'kuka_kr4r600::link_1'
        try:
            resp = self.set_link_target(link2_msg)
        except rospy.ServiceException as e:
            print("Set Link2 Target Service Failed: %s" % e)

        link3_msg = LinkState()
        link3_msg.link_name = 'kuka_kr4r600::link_3'
        link3_msg.pose.position.x = robot_init_pose[14]
        link3_msg.pose.position.y = robot_init_pose[15]
        link3_msg.pose.position.z = robot_init_pose[16]
        link3_msg.pose.orientation.x = robot_init_pose[17]
        link3_msg.pose.orientation.y = robot_init_pose[18]
        link3_msg.pose.orientation.z = robot_init_pose[19]
        link3_msg.pose.orientation.w = robot_init_pose[20]
        link3_msg.reference_frame = 'kuka_kr4r600::link_2'
        try:
            resp = self.set_link_target(link3_msg)
        except rospy.ServiceException as e:
            print("Set Link3 Target Service Failed: %s" % e)

        link4_msg = LinkState()
        link4_msg.link_name = 'kuka_kr4r600::link_4'
        link4_msg.pose.position.x = robot_init_pose[21]
        link4_msg.pose.position.y = robot_init_pose[22]
        link4_msg.pose.position.z = robot_init_pose[23]
        link4_msg.pose.orientation.x = robot_init_pose[24]
        link4_msg.pose.orientation.y = robot_init_pose[25]
        link4_msg.pose.orientation.z = robot_init_pose[26]
        link4_msg.pose.orientation.w = robot_init_pose[27]
        link4_msg.reference_frame = 'kuka_kr4r600::link_3'
        try:
            resp = self.set_link_target(link4_msg)
        except rospy.ServiceException as e:
            print("Set Link4 Target Service Failed: %s" % e)

        link5_msg = LinkState()
        link5_msg.link_name = 'kuka_kr4r600::link_5'
        link5_msg.pose.position.x = robot_init_pose[28]
        link5_msg.pose.position.y = robot_init_pose[29]
        link5_msg.pose.position.z = robot_init_pose[30]
        link5_msg.pose.orientation.x = robot_init_pose[31]
        link5_msg.pose.orientation.y = robot_init_pose[32]
        link5_msg.pose.orientation.z = robot_init_pose[33]
        link5_msg.pose.orientation.w = robot_init_pose[34]
        link5_msg.reference_frame = 'kuka_kr4r600::link_4'
        try:
            resp = self.set_link_target(link5_msg)
        except rospy.ServiceException as e:
            print("Set Link5 Target Service Failed: %s" % e)

        rospy.wait_for_service('gazebo/set_link_state')
        rospy.sleep(0.5)
        '''
        Transfer the initial robot state to the state for the DDPG Agent
        '''
        rob_state = self._get_next_robot_state()
        rospy.wait_for_service('gazebo/pause_physics')
        try:
            self.pause_gazebo()
        except rospy.ServiceException as e:
            print("Pause Service Failed: %s" % e)
        # goal_dis = self._compute_dis_2_goal(rob_state[0])
        self.goal_dis_pre = self.robot_goal_dist
        self.goal_dis_cur = self.robot_goal_dist
        self.scan_dist_pre = self.collision
        self.scan_dist_cur = self.collision
        state = self._robot_state_2_ddpg_state(rob_state)
        return state

    def reset_environment(self, init_pose_list):
        """
        Set New Environment for training
        :param init_pose_list: init pose list of robot
        :param goal_list: goal position list
        :param obstacle_list: obstacle list
        """
        self.robot_init_pose_list = init_pose_list


    def _get_next_robot_state(self):
        """
        Get the combination of state after execute the action for a certain time
        State will be: [robot_pose, robot_spd, col_dist]
        :return: state
        """
        tmp_robot_pose = copy.deepcopy(self.robot_pose)
        tmp_robot_speed = copy.deepcopy(self.robot_speed)
        # tmp_dist = self._compute_dis_2_goal(self.found_obj)
        tmp_goal_dist = copy.deepcopy(self.collision)
        state = [tmp_robot_pose, tmp_robot_speed, tmp_goal_dist]
        return state

    def _robot_state_2_ddpg_state(self, state):
        """
        Transform robot state to DDPG state
        Robot State: [robot_pose, robot_spd, scan]
        DDPG state: [Distance to goal from camera, Pose, Speed, collision_dist]
        :param state: robot state
        :return: ddpg_state
        """
        # State 0 include the pose(position and orientation) of all joints
        # State 1 include the speed(linear and angular) of all joints
        # State 2 include all distances
        ddpg_state = [self.goal_dis_cur, state[0], state[1], state[2]]
        return ddpg_state

    def _compute_reward(self, state):
        """
        Compute Reward of the action base on current DDPG state and 
        last step goal distance and direction

        Reward:
            1. R_Arrive If Distance to Goal is smaller than D_goal
            2. R_Collision If Distance to Obstacle is smaller than D_obs
            3. a * (Last step distance to goal - current step distance to goal)

        If robot near obstacle then done
        :param state: DDPG state
        :return: reward, done
        """
        done = False

        near_obstacle = False
        # First check distance from scanner if scan show that it is near obstacle

        if self.collision < 0.15 and self.found_obj == 0:
            print("near_obstacle")
            near_obstacle = True
            
        '''
        Assign Rewards
        '''

        if self.goal_dis_cur < self.goal_near_th:
            reward = self.goal_reward
            done = True
        elif near_obstacle:
            reward = self.col_reward
            done = True
        else:
            reward = self.goal_dis_amp * (self.goal_dis_pre - self.goal_dis_cur)
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
            
            # pose_j1 = np.concatenate((pos_j1, quat_j1))
            # pose_j2 = np.concatenate((pos_j2, quat_j2))
            # pose_j3 = np.concatenate((pos_j3, quat_j3))
            # pose_j4 = np.concatenate((pos_j4, quat_j4))
            # pose_j5 = np.concatenate((pos_j5, quat_j5))

            # vel_j1 = np.concatenate((linear_j1, angular_j1))
            # vel_j2 = np.concatenate((linear_j2, angular_j2))
            # vel_j3 = np.concatenate((linear_j3, angular_j3))
            # vel_j4 = np.concatenate((linear_j4, angular_j4))
            # vel_j5 = np.concatenate((linear_j5, angular_j5))

            # self.robot_pose = np.concatenate((pose_j1, pose_j2, pose_j3, pose_j4, pose_j5))
            # self.robot_speed = np.concatenate((vel_j1, vel_j2, vel_j3, vel_j4, vel_j5))

    def _robot_distance_cb(self,msg):
        if self.robot_distance_init is False:
            self.robot_distance_init = True
        self.robot_goal_dist = msg.data
        self.found_obj = msg.havered

    def collision_dist_cb(self,msg):
        if self.collision_dist_init is False:
            self.collision_dist_init = True
        self.collision = msg.data
