import rospy
import sys
from datetime import datetime
sys.path.append('../../')
from eval_simulation import RandEvalGpu
from utility import *
import os

def evaluate_ddpg(pos_start=0, pos_end=199, model_name='ddpg', save_dir="/home/katerina/ddpg_model/3_10/",
                  state_num=14, is_scale=True, is_poisson=True, is_save_result=True,
                  use_cuda=True):
    """
    Evaluate DDPG in Simulated Environment
    :param pos_start: Start index position for evaluation
    :param pos_end: End index position for evaluation
    :param model_name: name of the saved model
    :param save_dir: directory to the saved model
    :param state_num: shape of input state
    :param is_scale: if true normalize input state
    :param is_poisson: if true use Poisson encoding for input state
    :param is_save_result: if true save the evaluation result
    :param use_cuda: if true use gpu
    """
    rospy.init_node('ddpg_eval')
    # poly_list, raw_poly_list = gen_test_env_poly_list_env()
    # start_goal_pos = pickle.load(open("eval_positions.p", "rb"))
    # robot_init_list = [[0.1, -1.35, 1.9, 0.0, 0.61],
    #                     [0.0, -1.25, 1.9, 0.0, 0.61],[0.0, -2.5, 2.3, 0.0, 1.0],
    #                     [0.0, -2.0, 1.5, 0.0, 1.55], [0.0, -1.57, 0.6, 0.0, 2.09]]
    robot_init_list = [[0.3, -1.35, 1.9, 0.0, 0.61], [0.1, -1.35, 1.9, 0.0, 0.61],
                        [0.0, -1.25, 1.9, 0.0, 0.61], [0.1, -1.35, 1.9, 0.10, 0.61],
                        [0.0, -1.5, 1.5, 0.0, 0.2], [0.3, -1.5, 2.3, 0.20, 0.6],
                        [0.0, -2.0, 1.5, 0.0, 1.55], [0.0, -1.7, 2.2, 0.0, 1.7],
                        [0.0, -1.57, 0.6, 0.0, 2.09], [-0.10, -1.57, 1.6, -0.20, 2.09],
                        [0.10, -1.57, 1.3, 0.10, 2.09],[-0.30, -2.0, 0.6, -0.20, 1.2],
                        [0.0, -2.0, 1.5, 0.0, 1.55],[0.1, -1.25, 1.9, 0.10, 0.61],
                        [0.10, -1.35, 1.5, 0.0, 1.55],[0.1, -1.25, 1.9, -0.20, 0.61],
                        [0.0, -1.0, -0.50, 0.0, 1.55],[0.1, -1.8, 1.9, 0.10, 0.61],
                        [0.0, -2.0, 1.5, 0.0, 1.55],[0.1, -1.25, 1.9, -0.10, 0.61]
                        ]
    goal_list = [0.5, 0.0, 0.85]
    # net_dir = '/home/katerina/save_ddpg_weights_simple/10_10/_ddpg_actor_model_final_model.pt'
    net_dir = '/home/katerina/save_ddpg_weights_subgoals/6_10/_ddpg_actor_model_final_model.pt'
    for run_num in range(5):
        actor_net = load_test_actor_network(net_dir, state_num=state_num)
        eval = RandEvalGpu(actor_net, robot_init_list, goal_list,
                           max_steps=100, action_rand=0.01, is_spike=False,
                           is_scale=is_scale, is_poisson=is_poisson, use_cuda=use_cuda)
        data = eval.run_ros()
        save_at='../record_data/ddpg_subgoal_revision/'
        try:
            os.mkdir(save_at)
            print("Directory ", save_at, " Created")
        except FileExistsError:
            print("Directory", save_at, " already exists")
        pickle.dump(data,open(save_at+'/'+ 'episode_'+'{ep_num}'.format(ep_num=1) + '_run_'+'{run}'.format(run=run_num)+'.p', 'wb+'))
        print("Save at: "+ save_at+'episode_'+'{ep_num}'.format(ep_num=1) + '_run_'+'{run}'.format(run=run_num)+ " Eval on GPU Finished ...")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--poisson', type=int, default=0)
    args = parser.parse_args()

    USE_CUDA = True
    if args.cuda == 0:
        USE_CUDA = False
    IS_POISSON = False
    STATE_NUM = 14
    MODEL_NAME = 'ddpg'
    if args.poisson == 1:
        IS_POISSON = True
        STATE_NUM = 14
        MODEL_NAME = 'ddpg_poisson'
    SAVE_RESULT = False
    if args.save == 1:
        SAVE_RESULT = True
    evaluate_ddpg(use_cuda=USE_CUDA, state_num=STATE_NUM,
                  is_poisson=IS_POISSON, model_name=MODEL_NAME,
                  is_save_result=SAVE_RESULT)
