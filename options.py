import argparse
import os
import torch
import pathlib
from shutil import rmtree

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--gpu_id', type=int, default=0, help='gpu id: e.g. 0, 1. -1 is no GPU')

        self.parser.add_argument('--dataset', type=str, default='shapenet', help='shapenet')
        self.parser.add_argument('--dataroot', default='/ssd/dataset/shapenetcore_partanno_segmentation_benchmark_v0_normal/', help='path to images & laser point clouds')
        self.parser.add_argument('--classes', type=int, default=50, help='ModelNet40 or ModelNet10')
        self.parser.add_argument('--name', type=str, default='train', help='name of the experiment. It decides where to store samples and models')

        self.parser.add_argument('--checkpoints_dir', type=str, default=pathlib.Path(pathlib.Path(__file__).parent.absolute(), 'checkpoints'), help='models are saved here')

        self.parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        self.parser.add_argument('--input_pc_num', type=int, default=1024, help='# of input points')
        self.parser.add_argument('--surface_normal', type=bool, default=True, help='use surface normal in the pc input')
        self.parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')

        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=200, help='window id of the web display')

        self.parser.add_argument('--feature_num', type=int, default=1024, help='length of encoded feature')
        self.parser.add_argument('--activation', type=str, default='relu', help='activation function: relu, elu')
        self.parser.add_argument('--normalization', type=str, default='batch', help='normalization function: batch, instance')

        self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        self.parser.add_argument('--dropout', type=float, default=0.6, help='probability of an element to be zeroed')
        self.parser.add_argument('--node_num', type=int, default=64, help='som node number')
        self.parser.add_argument('--k', type=int, default=3, help='k nearest neighbor')
        # '/ssd/open-source/so-net-full/autoencoder/checkpoints/save/shapenetpart/183_0.034180_net_encoder.pth'
        self.parser.add_argument('--pretrain', type=str, default=None, help='pre-trained encoder dict path')
        self.parser.add_argument('--pretrain_lr_ratio', type=float, default=1, help='learning rate ratio between pretrained encoder and classifier')

        self.parser.add_argument('--som_k', type=int, default=9, help='k nearest neighbor of SOM nodes searching on SOM nodes')
        self.parser.add_argument('--som_k_type', type=str, default='center', help='avg / center')

        self.parser.add_argument('--random_pc_dropout_lower_limit', type=float, default=1, help='keep ratio lower limit')
        self.parser.add_argument('--bn_momentum', type=float, default=0.1, help='normalization momentum, typically 0.1. Equal to (1-m) in TF')
        self.parser.add_argument('--bn_momentum_decay_step', type=int, default=None, help='BN momentum decay step. e.g, 0.5->0.01.')
        self.parser.add_argument('--bn_momentum_decay', type=float, default=0.6, help='BN momentum decay step. e.g, 0.5->0.01.')


        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        self.opt.device = torch.device("cuda:%d" % (self.opt.gpu_id) if torch.cuda.is_available() else "cpu")
        # torch.cuda.set_device(self.opt.gpu_id)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir =  os.path.join(self.opt.checkpoints_dir, self.opt.name)
        if os.path.isdir(expr_dir):
            res = input('checkpoint dir %s is being replaced; are you sure? y/n' % expr_dir)
            if res == 'y':
                rmtree(expr_dir)
            elif res == 'n':
                exit('Please choose a different name for experiment using --name option.')
            else:
                exit('Please answer y/n.')
        os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
