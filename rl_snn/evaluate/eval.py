import pickle
import rospy
import sys
from datetime import datetime
sys.path.append('../../')
from eval_simulation import RandEvalGpu
from utility import *
import csv  
import glob
import pathlib
import os

def evaluate_sddpg(pos_start=0, pos_end=199, model_name='sddpg_bw_5', save_dir='/home/katerina/save_sddpg_subgoals/5_10/',
                   batch_window=5, is_save_result=True, use_cuda=True):
    """
    Evaluate Spiking DDPG in Simulated Environment
    :param pos_start: Start index position for evaluation
    :param pos_end: End index position for evaluation
    :param model_name: name of the saved model
    :param save_dir: directory to the saved model
    :param batch_window: inference timesteps
    :param is_save_result: if true save the evaluation result
    :param use_cuda: if true use gpu
    """
    rospy.init_node('sddpg_eval')
    # poly_list, raw_poly_list = gen_test_env_poly_list_env()
    start_goal_pos = [0.0, -1.35, 1.9, 0.0, 0.61]
    robot_init_list = [[0.0, -1.35, 1.9, 0.0, 0.61],[0.1, -1.35, 1.9, 0.0, 0.61],
                        [0.0, -1.25, 1.9, 0.0, 0.61],[0.0, -2.5, 2.3, 0.0, 1.0],
                        [0.0, -2.0, 1.5, 0.0, 1.55], [0.0, -1.57, 0.6, 0.0, 2.09]]
    goal_list = [0.5, 0.0, 0.85]
    weights = []
    bias = []
    for i in range(10,260,10):
        w_dir =  glob.glob(save_dir+'*_weights_s'+'{num}'.format(num=i)+'.p')
        for f in w_dir:
            weights.append(f)
        b_dir = glob.glob(save_dir+'*_bias_s'+'{num}'.format(num=i)+'.p')
        for f in b_dir:
            bias.append(f)
    episode = 0

    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    for w in range(len(weights)):
        episode +=1
        for run_num in range(5):
            actor_net = load_test_actor_snn_network(weights[w], bias[w], device, batch_window=batch_window)
            eval = RandEvalGpu(actor_net, robot_init_list, goal_list,
                               max_steps=100, action_rand=0.01,
                               is_spike=True, use_cuda=use_cuda)
            data = eval.run_ros()
            save_at='../record_data/snn_with_subgoals/5_10_b' 
            try:
                os.mkdir(save_at)
                print("Directory ", save_at, " Created")
            except FileExistsError:
                print("Directory", save_at, " already exists")
            pickle.dump(data,open(save_at+'/'+ 'episode_'+'{ep_num}'.format(ep_num=episode) + '_run_'+'{run}'.format(run=run_num)+'.p', 'wb+'))
            print("Save at: "+ save_at+'/'+'episode_'+'{ep_num}'.format(ep_num=episode) + '_run_'+'{run}'.format(run=run_num)+ " Eval on GPU Finished ...")



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--step', type=int, default=5)
    args = parser.parse_args()

    USE_CUDA = True
    if args.cuda == 0:
        USE_CUDA = False
    SAVE_RESULT = False
    if args.save == 1:
        SAVE_RESULT = True
    MODEL_NAME = 'sddpg_bw_' + str(args.step)
    evaluate_sddpg(use_cuda=USE_CUDA, model_name=MODEL_NAME,
                   is_save_result=SAVE_RESULT)