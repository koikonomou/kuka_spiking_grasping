import rospy
import math
import time
import copy
import random
import torch
import numpy as np
from std_msgs.msg import Float64, Int32

from shapely.geometry import Point
from gazebo_msgs.msg import ModelStates, ModelState
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.srv import SetModelState
from visualization_msgs.msg import Marker
from table_exploration.msg import Distance
from table_exploration.msg import Collision
from visualization_msgs.msg import MarkerArray
from gazebo_msgs.msg import LinkStates, LinkState , ModelState
import sys
sys.path.append('../')
from training.utility import *


class RandEvalGpu:
    """ Perform Random Evaluation on GPU """
    def __init__(self,
                 actor_net,
                 robot_init_pose_list,
                 goal_pos_list,
                 ros_rate=10,
                 max_steps=2000,
                 j1_max=2.97, j1_min=-2.97, j2_max=0.50, j2_min=-3.40, j3_max=2.62, j3_min=-2.01, j4_max=3.23, j4_min=-3.23, j5_max=2.09, j5_min=-2.09,
                 is_spike=False,
                 is_scale=False,
                 is_poisson=False,
                 batch_window=50,
                 action_rand=0.05,
                 use_cuda=True,
                 goal_th=0.1,
                 is_record=False):
        """
        :param actor_net: Actor Network
        :param robot_init_pose_list: robot init pose list
        :param goal_pos_list: goal position list
        :param obstacle_poly_list: obstacle list
        :param ros_rate: ros rate
        :param max_steps: max step for single goal
        :param min_spd: min wheel speed
        :param max_spd: max wheel speed
        :param is_spike: is using SNN
        :param is_scale: is scale DDPG state input
        :param is_poisson: is use rand DDPG state input
        :param batch_window: batch window of SNN
        :param action_rand: random of action
        :param scan_half_num: half number of scan points
        :param scan_min_dis: min distance of scan
        :param goal_dis_min_dis: min distance of goal distance
        :param goal_th: distance for reach goal
        :param obs_near_th: distance for obstacle collision
        :param use_cuda: if true use cuda
        :param is_record: if true record running data
        """
        self.actor_net = actor_net
        self.robot_init_pose_list = robot_init_pose_list
        self.goal_pos_list = goal_pos_list
        self.ros_rate = ros_rate
        self.max_steps = max_steps
        self.is_spike = is_spike
        self.is_scale = is_scale
        self.is_poisson = is_poisson
        self.batch_window = batch_window
        self.action_rand = action_rand
        self.use_cuda = use_cuda
        self.is_record = is_record
        self.record_data = []
        self.goal_th = goal_th
        self.j1_max = j1_max
        self.j1_min = j1_min
        self.j2_max = j2_max
        self.j2_min = j2_min
        self.j3_max = j3_max
        self.j3_min = j3_min
        self.j4_max = j4_max
        self.j4_min = j4_min
        self.j5_max = j5_max 
        self.j5_min = j5_min
        self.spike_state_num = 14
        self.robot_joint_state_init = False
        self.robot_state_init = False
        self.camera_cb_init = False
        self.scan_dist_init = False
        self.collision_init = False
        # Put network to device
        if self.use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.actor_net.to(self.device)
        # Robot State
        self.robot_state_init = False
        self.robot_scan_init = False
        self.robot_pose = [0., 0., 0., 0., 0., 0.]
        self.robot_spd = [0., 0., 0., 0., 0., 0.]
        # Subscriber
        # rospy.Subscriber('gazebo/model_states', ModelStates, self._robot_state_cb)
        # rospy.Subscriber('simplescan', SimpleScan, self._robot_scan_cb)
        # rospy.Subscriber('/gazebo/link_states', LinkStates, self._robot_link_cb )
        rospy.Subscriber('/kuka/collision', Float64, self.scan_dist_cb)
        rospy.Subscriber('/kuka/box/distance', Distance, self.camera_cb )
        rospy.Subscriber('/collision_detection', Collision, self.collision_cb)
        rospy.Subscriber('/joint_states', JointState, self.joint_state_cb)
        # Publisher

        # Publisher
        self.pub_a1 = rospy.Publisher('/kuka_kr4r600/joint_a1_position_controller/command', Float64, queue_size=1)
        self.pub_a2 = rospy.Publisher('/kuka_kr4r600/joint_a2_position_controller/command', Float64, queue_size=1)
        self.pub_a3 = rospy.Publisher('/kuka_kr4r600/joint_a3_position_controller/command', Float64, queue_size=1)
        self.pub_a4 = rospy.Publisher('/kuka_kr4r600/joint_a4_position_controller/command', Float64, queue_size=1)
        self.pub_a5 = rospy.Publisher('/kuka_kr4r600/joint_a5_position_controller/command', Float64, queue_size=1)
        # Service
        self.set_model_target = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        # Init Subscriber
        # while not self.robot_state_init:
        #     continue
        while not self.scan_dist_init:
            continue
        while not self.camera_cb_init:
            continue
        while not self.collision_init:
            continue
        while not self.robot_joint_state_init:
            continue
        rospy.loginfo("Finish Subscriber Init...")

    def run_ros(self):
        """
        ROS ROS Node
        :return: run_data
        """
        run_num = 1
        run_data = {"final_state": np.zeros(run_num),
                    "time": np.zeros(run_num),
                    "path": []}
        rate = rospy.Rate(self.ros_rate)
        goal_ita = 0
        single_goal_run_ita = 0
        failure_case = 0
        robot_path = []
        # self._set_new_target(goal_ita)
        print("Test: ", goal_ita)
        print("Start Robot Pose: (%.3f, %.3f, %.3f, %.3f, %.3f) Goal: (%.3f, %.3f, %.3f)" %
              (self.robot_init_pose_list[0], self.robot_init_pose_list[1],self.robot_init_pose_list[2],
               self.robot_init_pose_list[3],self.robot_init_pose_list[4],
               self.goal_pos_list[0], self.goal_pos_list[1], self.goal_pos_list[2]))
        goal_start_time = time.time()
        while not rospy.is_shutdown():
            tmp_robot_pose = copy.deepcopy(self.robot_pose)
            tmp_robot_spd = copy.deepcopy(self.robot_spd)
            tmp_robot_distance = copy.deepcopy(self.actual_dist)
            tmp_robot_distance = np.clip(tmp_robot_distance, 0, 1)
            # is_near_obs = self._near_obstacle(tmp_robot_pose)
            robot_path.append(tmp_robot_pose)
            '''
            Set new test goal
            '''
            if self.actual_dist < self.goal_th or self.have_collide == 1 or single_goal_run_ita == self.max_steps:
                goal_end_time = time.time()
                run_data['time'][goal_ita] = goal_end_time - goal_start_time
                if self.actual_dist < self.goal_th:
                    print("End: Success")
                    run_data['final_state'][goal_ita] = 1
                elif self.have_collide:
                    failure_case += 1
                    print("End: Obstacle Collision")
                    run_data['final_state'][goal_ita] = 2
                elif single_goal_run_ita == self.max_steps:
                    failure_case += 1
                    print("End: Out of steps")
                    run_data['final_state'][goal_ita] = 3
                print("Up to step failure number: ", failure_case)
                run_data['path'].append(robot_path)
                goal_ita += 1
                if goal_ita == run_num:
                    break
                single_goal_run_ita = 0
                robot_path = []
                print("Test: ", goal_ita)
                print("Start Robot Pose: (%.3f, %.3f, %.3f, %.3f, %.3f ) Goal: (%.3f, %.3f, %.3f)" %
                      (self.robot_init_pose_list[0], self.robot_init_pose_list[1],self.robot_init_pose_list[2],
                       self.robot_init_pose_list[3],self.robot_init_pose_list[4],
                       self.goal_pos_list[0], self.goal_pos_list[1],self.goal_pos_list[2]))
                goal_start_time = time.time()
                continue
            '''
            Perform Action
            '''
            # tmp_goal_dis = goal_dis
            # if tmp_goal_dis == 0:
            #     tmp_goal_dis = 1
            # else:
            #     tmp_goal_dis = self.goal_dis_min_dis / tmp_goal_dis
            #     if tmp_goal_dis > 1:
            #         tmp_goal_dis = 1
            ddpg_state = [tmp_robot_distance, tmp_robot_pose, tmp_robot_spd, self.found_obj]
            # ddpg_state.extend(tmp_robot_scan.tolist())
            action = self._network_2_robot_action(ddpg_state)
            move_joint_a1 = Float64()
            move_joint_a2 = Float64()
            move_joint_a3 = Float64()
            move_joint_a4 = Float64()
            move_joint_a5 = Float64()
            move_joint_a1.data = action[0]
            move_joint_a2.data = action[1]
            move_joint_a3.data = action[2]
            move_joint_a4.data = action[3]
            move_joint_a5.data = action[4]
            self.pub_a1.publish(move_joint_a1)
            self.pub_a2.publish(move_joint_a2)
            self.pub_a3.publish(move_joint_a3)
            self.pub_a4.publish(move_joint_a4)
            self.pub_a5.publish(move_joint_a5)
            single_goal_run_ita += 1
            rate.sleep()
        suc_num = np.sum(run_data["final_state"] == 1)
        obs_num = np.sum(run_data["final_state"] == 2)
        out_num = np.sum(run_data["final_state"] == 3)
        print("Success: ", suc_num, " Obstacle Collision: ", obs_num, " Over Steps: ", out_num)
        print("Success Rate: ", suc_num / run_num)
        return run_data

    def _network_2_robot_action(self, state):
        """
        Generate robot action based on network output
        :param state: ddpg state
        :return: [linear spd, angular spd]
        """
        # state = np.array(state).reshape((-1))

        with torch.no_grad():
            if self.is_spike:
                state = self._state_2_state_spikes(state)
                if self.is_record:
                    self.record_data.append(state)
                state = torch.Tensor(state).to(self.device)
                action = self.actor_net(state, 1).to('cpu')
            elif self.is_scale:
                state = self._state_2_scale_state(state)
                if self.is_record:
                    self.record_data.append(state)
                state = torch.Tensor(state).to(self.device)
                action = self.actor_net(state).to('cpu')
            else:
                state = np.array(state).reshape((1, -1))
                if self.is_record:
                    self.record_data.append(state)
                state = torch.Tensor(state).to(self.device)
                action = self.actor_net(state).to('cpu')
            action = action.numpy().squeeze()
        noise = np.random.randn(5) * self.action_rand
        action = noise + (1 - self.action_rand) * action
        action = np.clip(action, [0., 0., 0., 0., 0.], [1., 1., 1., 1., 1.])
        action = network_2_robot_action_decoder(action, self.j1_max, self.j1_min, self.j2_max, self.j2_min, 
            self.j3_max, self.j3_min, self.j4_max, self.j4_min, self.j5_max, self.j5_min)
        return action
    def flatten_list(self, _2d_list):
        flat_list = []
        # Iterate through the outer list
        for element in _2d_list:
            if type(element) is list:
                # If the element is of type list, iterate through the sublist
                for item in element:
                    flat_list.append(item)
            else:
                flat_list.append(element)
        return flat_list

    def _state_2_state_spikes(self, state):
        """
        Transform state to spikes of input neurons
        :param state: robot state
        :return: state_spikes
        """
        spike_state_num = self.spike_state_num

        # spike_state_value = snn_state_2_spike_value_state(state, spike_state_num)
        spike_state_value = self.flatten_list(state)
        spike_state_value = np.array(spike_state_value)

        # spike_state_value = np.array(state)
        spike_state_value = spike_state_value.reshape((-1, spike_state_num, 1))
        state_spikes = np.random.rand(1, spike_state_num, self.batch_window) < spike_state_value
        state_spikes = state_spikes.astype(float)
        return state_spikes

    def _state_2_scale_state(self, state):
        """
        Transform state to scale state with or without Poisson random
        :param state: robot state
        :return: scale_state
        """
        if self.is_poisson:
            scale_state_num = self.spike_state_num
            state = snn_state_2_spike_value_state(state, scale_state_num)
            state = np.array(state).reshape((-1))
            # state = np.array(state)
            spike_state_value = state.reshape((-1, scale_state_num, 1))
            state_spikes = np.random.rand(1, scale_state_num, self.batch_window) < spike_state_value
            poisson_state = np.sum(state_spikes, axis=2).reshape((1, -1))
            poisson_state = poisson_state / self.batch_window
            scale_state = poisson_state.astype(float)
        else:
            scale_state_num = self.scan_half_num * 2 + 4
            # scale_state = ddpg_state_rescale(state, scale_state_num)
            scale_state = np.array(scale_state).reshape((1, scale_state_num))
            scale_state = scale_state.astype(float)
        return scale_state

    # def _near_obstacle(self, pos):
    #     """
    #     Test if robot is near obstacle
    #     :param pos: robot position
    #     :return: done
    #     """
    #     done = False
    #     robot_point = Point(pos[0], pos[1])
    #     for poly in self.obstacle_poly_list:
    #         tmp_dis = robot_point.distance(poly)
    #         if tmp_dis < self.obs_near_th:
    #             done = True
    #             break
    #     return done

    # def _set_new_target(self, ita):
        # """
        # Set new robot pose and goal position
        # :param ita: goal ita
        # """
        # goal_position = self.goal_pos_list[ita]
        # target_msg = ModelState()
        # target_msg.model_name = 'target'
        # target_msg.pose.position.x = goal_position[0]
        # target_msg.pose.position.y = goal_position[1]
        # rospy.wait_for_service('gazebo/set_model_state')
        # try:
        #     resp = self.set_model_target(target_msg)
        # except rospy.ServiceException as e:
        #     print("Set Target Service Failed: %s" % e)
        # self.pub_action.publish(Twist())
        # robot_init_pose = self.robot_init_pose_list[ita]
        # robot_init_quat = self._euler_2_quat(yaw=robot_init_pose[2])
        # robot_msg = ModelState()
        # robot_msg.model_name = 'mobile_base'
        # robot_msg.pose.position.x = robot_init_pose[0]
        # robot_msg.pose.position.y = robot_init_pose[1]
        # robot_msg.pose.orientation.x = robot_init_quat[1]
        # robot_msg.pose.orientation.y = robot_init_quat[2]
        # robot_msg.pose.orientation.z = robot_init_quat[3]
        # robot_msg.pose.orientation.w = robot_init_quat[0]
        # rospy.wait_for_service('gazebo/set_model_state')
        # try:
        #     resp = self.set_model_target(robot_msg)
        # except rospy.ServiceException as e:
        #     print("Set Target Service Failed: %s" % e)
        # rospy.sleep(0.5)



    # def _robot_link_cb(self, msg):
    #     """
    #     Callback function for robot state
    #     :param msg: message
    #     """
    #     if self.robot_state_init is False:
    #         self.robot_state_init = True
    #     quat = [msg.pose[-1].orientation.x,
    #             msg.pose[-1].orientation.y,
    #             msg.pose[-1].orientation.z,
    #             msg.pose[-1].orientation.w]
    #     siny_cosp = 2. * (quat[0] * quat[1] + quat[2] * quat[3])
    #     cosy_cosp = 1. - 2. * (quat[1] ** 2 + quat[2] ** 2)
    #     yaw = math.atan2(siny_cosp, cosy_cosp)
    #     linear_spd = math.sqrt(msg.twist[-1].linear.x**2 + msg.twist[-1].linear.y**2)
    #     self.robot_pose = [msg.pose[-1].position.x, msg.pose[-1].position.y, yaw]
    #     self.robot_spd = [linear_spd, msg.twist[-1].angular.z]
    
    def joint_state_cb(self,msg):
        if self.robot_joint_state_init is False:
            self.robot_joint_state_init = True
        self.robot_pose = [msg.position[0],msg.position[1],msg.position[2],msg.position[3],msg.position[4],msg.position[5]]
        self.robot_spd = [msg.velocity[0],msg.velocity[1],msg.velocity[2],msg.velocity[3],msg.velocity[4],msg.velocity[5]]
    
    # def _robot_scan_cb(self, msg):
    #     """
    #     Callback function for robot laser scan
    #     :param msg: message
    #     """
    #     if self.robot_scan_init is False:
    #         self.robot_scan_init = True
    #     tmp_robot_scan_ita = 0
    #     for num in range(self.scan_half_num):
    #         ita = self.scan_half_num - num - 1
    #         self.robot_scan[tmp_robot_scan_ita] = msg.data[ita]
    #         tmp_robot_scan_ita += 1
    #     for num in range(self.scan_half_num):
    #         ita = len(msg.data) - num - 1
    #         self.robot_scan[tmp_robot_scan_ita] = msg.data[ita]
    #         tmp_robot_scan_ita += 1
    
    def collision_cb(self,msg):
        if self.collision_init is False:
            self.collision_init = True
        self.have_collide = msg.have_collide
        self.actual_dist = msg.goal_dist
    def camera_cb(self,msg):
        if self.camera_cb_init is False:
            self.camera_cb_init = True
    # self.robot_goal_dist = msg.data
        self.found_obj = msg.havered

    def scan_dist_cb(self,msg):
        if self.scan_dist_init is False:
            self.scan_dist_init = True
        self.scan_dist = msg.data
