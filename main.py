# -*- coding: utf-8 -*-
from __future__ import print_function, division
import nsml
import torch
import utils
import model_all
import argparse
import train
import copy
import pickle
import subprocess
import os
from nsml import DATASET_PATH

def get_args():
    model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vgg16', 'densenet169']
    # model_names = sorted(name for name in models.__dict__
    #                      if name.islower() and not name.startswith("__") and name.startswith("resnet"))

    parser = argparse.ArgumentParser(description='Vincent Evaluator Training')
    parser.add_argument('--adaptive_pool', action='store_true', default=False)
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet101',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--data_type', type=str, default='c')
    #parser.add_argument('--depth', type=int, default=101)
    parser.add_argument('--description', type=str, default='binary_class')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--fc_num', type=int, default=2)
    parser.add_argument('--k_fold', type=int, default=0)
    parser.add_argument('--k_fold_order', type=int, default=0)
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_sch', type=int, default=0)
    parser.add_argument('--method', type=str, default='classification')
    parser.add_argument('--milestones', type=int,nargs='+')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--open_num', type=int, default=4)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--short_data', action='store_true', default=False)
    parser.add_argument('--target_data', type=str, default='ILSVRC')
    parser.add_argument('--test_tf', type=str, default='[transforms.Resize((255, 255)),transforms.CenterCrop((224, 224))]')
    parser.add_argument('--train_tf', type=str, default='[transforms.Resize((255, 255)),transforms.RandomCrop((224, 224))]')
    parser.add_argument('--use_dropout', type=int, default=0)
    #parser.add_argument('--use_dropout', action='store_true', default=False)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--use_pretrained', action='store_true', default=False)
    parser.add_argument('--use_multicrop', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--softNDCG_sigma',type = int, default = 0)
    parser.add_argument('--softNDCG_at',type = int, default = -1)
    parser.add_argument('--subset',action= 'store_true',default = False)
    parser.add_argument('--whole_batch',action= 'store_true',default = False)
    args = parser.parse_args()
    args.use_gpu = torch.cuda.is_available()
    return args

def infer(input):
    if args.method == 'regression':
        return utils.infer_regression(input, model)
    else:
        return utils.infer_classification(input, model)


def main():
    args = get_args()
    if args.use_dropout == 0:
        args.use_dropout = False

    if args.use_dropout ==0:
        args.use_dropout = False

    for x in vars(args).items():
        print(x)
    #from utils import data_transforms
    #print(data_transforms)

    if args.lr_sch ==5 and torch.__version__ != '0.4.0' :
        print("for cosine annealing, change to torch==0.4.0 in setup.py")
        raise AssertionError()
    elif args.lr_sch !=5 and torch.__version__ == '0.4.0':
        print("warning : this is torch version {}! nsml report will not be recorded".format(torch.__version__))


    model, optimizer, scheduler = model_all.get_model(args)

    if args.use_gpu:
        if torch.cuda.device_count() > 1:
            print("[gpu] Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = torch.nn.DataParallel(model)
        elif torch.cuda.device_count() == 1:
            print("[gpu] Let's use", torch.cuda.device_count(), "GPUs!")
        else:
            print("[gpu] no available gpus")
        model = model.cuda()
    

    nsml.bind(infer=infer, model=model, optimizer=optimizer)

    if args.pause:
        nsml.paused(scope=locals())

    nsml.save()
    if args.mode == 'train':
        dataloaders, dataset_sizes = utils.data_loader(args, train=True, batch_size=args.batch_size)
        model = train.train_test(model, optimizer, scheduler, dataloaders, dataset_sizes, args)
    
    utils.save_model(model, 'model_state')
    with open('args.pickle', 'wb') as farg:
        pickle.dump(args, farg)

    loader = utils.data_loader(args, train=False, batch_size=1)
    predict, acc = utils.get_forward_result(model, loader, args)
    predict = torch.cat(predict, 0)
    nsml.bind(save=lambda x: utils.save_csv(x,
                                            data_csv_fname=os.path.join(DATASET_PATH, 'train', 'test') + '/test_data',
                                            results=predict,
                                            test_loader=loader))
    nsml.save('result')
    
if __name__ == '__main__':
    main()

#if __name__ == '__main__':
#    import cProfile
#    cProfile.run('main()', 'main_stats')