import os
import argparse
from solver import Solver
import model
import pickle
#from douban_LoadData import LoadData
from LoadData import LoadData
import pdb

def pars_args():

    parser = argparse.ArgumentParser() 
    
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dataset_type', type=str, default=None)
    parser.add_argument('--features',
        help='Which features to include.',
        choices=['none', 'categories', 'time', 'content', 'geo'],
        default='content')

    # Model hyper-parameters
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--N', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--h', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=32)
    parser.add_argument('--dropout', type=int, default=0.1)
    parser.add_argument('--std', type=float, default=0.01)
    parser.add_argument('--l2_reg', type=float, default=1e-4)
    parser.add_argument('--K', type=int, default=8)

    # Training settings
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=25)

    # Directories
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--model_save_dir', type=str, default='./models')

    # Step size
    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--model_save_step', type=int, default=50000)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = pars_args()

    data = LoadData('../dataset/', args.dataset, label='rating', sep=',', append_id=True, include_id=False)
    print(args)
    solver = Solver(data, args)
    solver.train()


