import time
import copy
import numpy as np
import math

from options import Options
opt = Options().parse()  # set CUDA_VISIBLE_DEVICES before import torch

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

from models import losses
from models.segmenter import Model
from data.intrA_loader import ShapeNetLoader
from data.intrA_loader import load_file_names
from util.visualizer import Visualizer
from sklearn.model_selection import KFold

runs_path = '/content/drive/MyDrive/code/SO-Net/runs'

if __name__=='__main__':
    
    filenames = load_file_names(opt.dataroot, 'file_name_list.txt')
    # k=5 split, 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True)

    # k-fold validation goes here, the outer-most loop
    # indexes = (train_index, test_index)
    for i, indexes in enumerate(kf.split(filenames)):
        
        # Only train 1 model for now
        if i > 0:
            break;
        
        train_indexes = indexes[0]
        test_indexes = indexes[1]
        train_file_names = filenames[train_indexes]
        test_file_names = filenames[test_indexes]

        trainset = ShapeNetLoader(train_file_names, 'train', opt)
        dataset_size = len(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)
        print('#training point clouds = %d' % len(trainset))

        testset = ShapeNetLoader(test_file_names, 'test', opt)
        testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)

        writer = SummaryWriter(runs_path)
        #visualizer = Visualizer(opt)

        # create model, optionally load pre-trained model
        model = Model(opt)
        if opt.pretrain is not None:
            model.encoder.load_state_dict(torch.load(opt.pretrain))

        # load pre-trained model
        # folder = 'checkpoints/'
        # model_epoch = '2'
        # model_acc = '0.914946'
        # model.encoder.load_state_dict(torch.load(folder + model_epoch + '_' + model_acc + '_net_encoder.pth'))
        # model.segmenter.load_state_dict(torch.load(folder + model_epoch + '_' + model_acc + '_net_segmenter.pth'))

        best_iou = 0
        for epoch in range(401):

            epoch_iter = 0
            print("Starting epoch %d..." % epoch)
            for i, data in enumerate(trainloader):
                iter_start_time = time.time()
                epoch_iter += opt.batch_size

                input_pc, input_sn, input_label, input_seg, input_node, input_node_knn_I = data
                model.set_input(input_pc, input_sn, input_label, input_seg, input_node, input_node_knn_I)

                model.optimize()

                if i % 100 == 0:
                    # print/plot errors
                    t = (time.time() - iter_start_time) / opt.batch_size

                    errors = model.get_current_errors()

                    writer.add_scalar("train_loss_seg", errors["train_loss_seg"], epoch)
                    writer.add_scalar("train_accuracy_seg", errors["train_accuracy_seg"], epoch)
                    writer.add_scalar("test_loss_seg", errors["test_loss_seg"], epoch)
                    writer.add_scalar("test_acc_seg", errors["test_acc_seg"], epoch)
                    writer.add_scalar("test_iou", errors["test_iou"], epoch)

                    #visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                    #visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

                    # print(model.autoencoder.encoder.feature)
                    # visuals = model.get_current_visuals()
                    # visualizer.display_current_results(visuals, epoch, i)

            # test network and visualize
            if epoch >= 0 and epoch%1==0:
                batch_amount = 0
                model.test_loss_segmenter.data.zero_()
                model.test_accuracy_segmenter.data.zero_()
                model.test_iou.data.zero_()
                for i, data in enumerate(testloader):
                    input_pc, input_sn, input_label, input_seg, input_node, input_node_knn_I = data
                    model.set_input(input_pc, input_sn, input_label, input_seg, input_node, input_node_knn_I)
                    model.test_model()

                    batch_amount += input_label.size()[0]

                    # # accumulate loss
                    model.test_loss_segmenter += model.loss_segmenter.detach() * input_label.size()[0]

                    _, predicted_seg = torch.max(model.score_segmenter.data, dim=1, keepdim=False)
                    correct_mask = torch.eq(predicted_seg, model.input_seg).float()
                    test_accuracy_segmenter = torch.mean(correct_mask)
                    model.test_accuracy_segmenter += test_accuracy_segmenter * input_label.size()[0]

                    # segmentation iou
                    test_iou_batch = losses.compute_iou(model.score_segmenter.cpu().data, model.input_seg.cpu().data, model.input_label.cpu().data, opt, input_pc.cpu().data)
                    model.test_iou += test_iou_batch * input_label.size()[0]

                    # print(test_iou_batch)
                    # print(model.score_segmenter.size())

                model.test_loss_segmenter /= batch_amount
                model.test_accuracy_segmenter /= batch_amount
                model.test_iou /= batch_amount
                if model.test_iou.item() > best_iou:
                    best_iou = model.test_iou.item()
                print('Tested network. So far best segmentation: %f' % (best_iou) )

                # save network
                if model.test_iou.item() > 0.835:
                    print("Saving network...")
                    model.save_network(model.encoder, 'encoder', '%d_%f' % (epoch, model.test_iou.item()), opt.gpu_id)
                    model.save_network(model.segmenter, 'segmenter', '%d_%f' % (epoch, model.test_iou.item()), opt.gpu_id)

            # learning rate decay
            if epoch%30==0 and epoch>0:
                model.update_learning_rate(0.5)

            # save network
            # if epoch%20==0 and epoch>0:
            #     print("Saving network...")
            #     model.save_network(model.classifier, 'cls', '%d' % epoch, opt.gpu_id)
        
        writer.flush()
        writer.close()





