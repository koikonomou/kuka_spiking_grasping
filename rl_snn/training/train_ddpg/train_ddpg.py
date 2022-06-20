
import rospy
import time
import os
from torch.utils.tensorboard import SummaryWriter
import sys
import pickle

sys.path.append('../../')
from training.train_ddpg.ddpg_agent import Agent
from training.training_env import GazeboEnvironment
from training.utility import *


def train_ddpg(run_name="SNN_R1", episode_num=100,
                iteration_num_start=200, iteration_num_step=1,
                iteration_num_max=1000,
                j1_max=2.97, j1_min=-2.97, j2_max=0.50, j2_min=-3.40, j3_max=2.62, j3_min=-2.01, j4_max=3.23, j4_min=-3.23, j5_max=2.09, j5_min=-2.09, save_steps=10000,
                env_epsilon=0.9, env_epsilon_decay=0.999,
                goal_dis_min=0.1,
                obs_reward=-20, goal_reward=10, goal_dis_amp=5, goal_th=0.5, obs_th=0.35,
                state_num=4, action_num=5, is_pos_neg=False, is_poisson=False, poisson_win=50, spike_state_num=68, batch_window=4, actor_lr=1e-5,
                memory_size=100000, batch_size=256, epsilon_end=0.1, rand_start=10000, rand_decay=0.999,
                rand_step=2, target_tau=0.01, target_step=1, use_cuda=True):
    """
    Training DDPG for Mapless Navigation

    :param run_name: Name for training run
    :param exp_name: Name for experiment run to get random positions
    :param episode_num: number of episodes for each of the 4 environments
    :param iteration_num_start: start number of maximum steps for 4 environments
    :param iteration_num_step: increase step of maximum steps after each episode
    :param iteration_num_max: max number of maximum steps for 4 environments
    :param linear_spd_max: max wheel speed
    :param linear_spd_min: min wheel speed
    :param save_steps: number of steps to save model
    :param env_epsilon: start epsilon of random action for 4 environments
    :param env_epsilon_decay: decay of epsilon for 4 environments
    :param laser_half_num: half number of scan points
    :param laser_min_dis: min laser scan distance
    :param scan_overall_num: overall number of scan points
    :param goal_dis_min_dis: minimal distance of goal distance
    :param obs_reward: reward for reaching obstacle
    :param goal_reward: reward for reaching goal
    :param goal_dis_amp: amplifier for goal distance change
    :param goal_th: threshold for near a goal
    :param obs_th: threshold for near an obstacle
    :param state_num: number of state
    :param action_num: number of action
    :param is_pos_neg: if true use two input neurons to represent position and negative
    :param is_poisson: if true use Poisson encoding
    :param poisson_win: Poisson encoding window size
    :param memory_size: size of memory
    :param batch_size: batch size
    :param epsilon_end: min value for random action
    :param rand_start: max value for random action
    :param rand_decay: steps from max to min random action
    :param rand_step: steps to change
    :param target_tau: update rate for target network
    :param target_step: number of steps between each target update
    :param use_cuda: if true use gpu
    """
    # Create Folder to save weights
    dirName = 'save_ddpg_weights'
    try:
        os.mkdir('../' + dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")

    # Define 4 training environments
    # Read Random Start Pose and Goal Position based on experiment name
    overall_init_list = [0.0,0.0,0.0,0.0,0.0]

    # Define Environment and Agent Object for training
    rospy.init_node("train_ddpg")
    env = GazeboEnvironment(goal_dis_min=goal_dis_min,
                            col_reward=obs_reward, goal_reward=goal_reward, goal_dis_amp=goal_dis_amp,
                            goal_near_th=goal_th, obs_near_th=obs_th)
    if is_pos_neg:
        rescale_state_num = state_num + 2
    else:
        rescale_state_num = state_num
    agent = Agent(state_num, action_num, rescale_state_num, poisson_window=poisson_win, use_poisson=is_poisson,
                  memory_size=memory_size, batch_size=batch_size, epsilon_end=epsilon_end,
                  epsilon_rand_decay_start=rand_start, epsilon_decay=rand_decay, epsilon_rand_decay_step=rand_step,
                  target_tau=target_tau, target_update_steps=target_step, use_cuda=use_cuda)

    # Define Tensorboard Writer
    tb_writer = SummaryWriter()

    # Define maximum steps per episode and reset maximum random action
    overall_steps = 0
    overall_episode = 0
    env_episode = 0
    env_ita = 0
    ita_per_episode = iteration_num_start

    env.reset_environment(overall_init_list)

    agent.reset_epsilon(env_epsilon,
                        env_epsilon_decay)

    # Start Training
    start_time = time.time()
    while True:
        state = env.reset(env_episode)
        if is_pos_neg:
            rescale_state = ddpg_state_2_spike_value_state(state, rescale_state_num)
        else:
            rescale_state = ddpg_state_rescale(state, rescale_state_num)
        episode_reward = 0
        for ita in range(ita_per_episode):
            ita_time_start = time.time()
            overall_steps += 1
            raw_action = agent.act(rescale_state)
            decode_action = network_2_robot_action_decoder(
                raw_action, j1_max, j1_min, j2_max, j2_min, j3_max, j3_min, j4_max, j4_min, j5_max, j5_min)
            next_state, reward, done = env.step(decode_action)
            if is_pos_neg:
                rescale_next_state = ddpg_state_2_spike_value_state(next_state, rescale_state_num)
            else:
                rescale_next_state = ddpg_state_rescale(state, rescale_state_num)

            # Add a last step negative reward
            episode_reward += reward
            agent.remember(state, rescale_state, raw_action, reward, next_state, rescale_next_state, done)
            state = next_state
            rescale_state = rescale_next_state

            # Train network with replay
            if len(agent.memory) > batch_size:
                actor_loss_value, critic_loss_value = agent.replay()
                tb_writer.add_scalar('DDPG/actor_loss', actor_loss_value, overall_steps)
                tb_writer.add_scalar('DDPG/critic_loss', critic_loss_value, overall_steps)
            ita_time_end = time.time()

            # Save Model
            if overall_steps % save_steps == 0:
                agent.save("../save_ddpg_weights", overall_steps // save_steps, run_name)

            # If Done then break
            if done or ita == ita_per_episode - 1:
                print("Episode: {}/{}, Avg Reward: {}, Steps: {}"
                      .format(overall_episode, episode_num, episode_reward / (ita + 1), ita + 1))
                tb_writer.add_scalar('DDPG/avg_reward', episode_reward / (ita + 1), overall_steps)
                break
        if ita_per_episode < iteration_num_max:
            ita_per_episode += iteration_num_step
        if overall_episode == 999:
            agent.save("../save_ddpg_weights", 0, run_name)
        overall_episode += 1
        env_episode += 1
        if env_episode == episode_num[env_ita]:
            print("Environment ", env_ita, " Training Finished ...")
            if env_ita == 3:
                break
            env_ita += 1
            env.reset_environment(overall_init_list)

            agent.reset_epsilon(env_epsilon,
                                env_epsilon_decay)
            ita_per_episode = iteration_num_start
            env_episode = 0
    end_time = time.time()
    print("Finish Training with time: ", (end_time - start_time) / 60, " Min")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--poisson', type=int, default=0)
    args = parser.parse_args()

    USE_CUDA = True
    if args.cuda == 0:
        USE_CUDA = False
    IS_POS_NEG, IS_POISSON = False, False
    if args.poisson == 1:
        IS_POS_NEG, IS_POISSON = True, True
    train_ddpg(use_cuda=USE_CUDA, is_pos_neg=IS_POS_NEG, is_poisson=IS_POISSON)
