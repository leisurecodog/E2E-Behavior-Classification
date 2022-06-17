import argparse

from yaml import parse

def add_parser():
    parser = argparse.ArgumentParser()
    """LSTM Training Param"""
    parser.add_argument('--LSTM_epochs', default=3000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--map_batch', default=8, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--obs_len', default=10, type=int)
    parser.add_argument('--f_len', default=1, type=int)
    
    parser.add_argument('--Reward_epochs', default=300, type=int)
    parser.add_argument('--state_num', default=120, type=int)
    parser.add_argument('--lstm_dataset', default='apolloScape')
    parser.add_argument('--feature_dataset', default='bdd100k')
    
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--save_img', default=False, action='store_true')
    parser.add_argument('--train_LSTM', default=False, action='store_true')
    parser.add_argument('--train_Reward', default=False, action='store_true')

    return parser.parse_args() 
