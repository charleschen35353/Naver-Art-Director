import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models


class FineTuneModel(nn.Module):
    def __init__(self, original_model, args):
        super(FineTuneModel, self).__init__()
        self.args = args
        if args.method == 'regression':
            num_classes = 1
        else:
            num_classes = args.class_num

        if args.arch.startswith('resnet'):
            print("[model] using resent network")
            self.feature1 = nn.Sequential(*list([original_model.conv1,
                                                 original_model.bn1,
                                                 original_model.relu,
                                                 original_model.maxpool,
                                                 original_model.layer1]))

            self.feature2 = original_model.layer2
            self.feature3 = original_model.layer3
            self.feature4 = original_model.layer4

            if args.adaptive_pool:
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
            else:
                self.avg_pool = nn.AvgPool2d(7, stride=2)


            if args.fc_num == 1:
                print("[model] using 1 fc layer")
                if args.data_type == 'c':
                    if args.use_dropout:
                        self.classifier = nn.Sequential(
                            nn.Dropout(0.5),
                            nn.Linear(original_model.fc.in_features, num_classes),
                            nn.Softmax()
                        )
                    else:
                        self.classifier = nn.Sequential(
                            nn.Linear(original_model.fc.in_features, num_classes),
                            nn.Softmax()
                        )
                elif args.data_type == 'cu':
                    if args.use_dropout:
                        self.classifier = nn.Sequential(
                            nn.Dropout(0.5),
                            nn.Linear(original_model.fc.in_features + 10 + 1, num_classes),
                            nn.Softmax()
                        )
                    else:
                        self.classifier = nn.Sequential(
                            nn.Linear(original_model.fc.in_features + 10 + 1, num_classes),
                            nn.Softmax()
                        )

                else:
                    print("[model] not proper data_type argument")
                    raise AssertionError()

            elif args.fc_num == 2:
                print("[model] using 2 fc layers")
                if args.data_type == 'c':
                    if args.use_dropout:
                        if args.use_dropout == 1:
                            print("[model] no user_feature 2 fc_num with only one first dropout in resnet")
                            self.classifier = nn.Sequential(
                                nn.Dropout(0.5),
                                nn.Linear(original_model.fc.in_features, original_model.fc.in_features // 4),
                                nn.ReLU(),
                                nn.Linear(original_model.fc.in_features // 4, num_classes),
                                nn.Softmax()
                            )
                        elif args.use_dropout == 2:
                            print("[model] no user_feature 2 fc_num with only one second dropout in resnet")
                            self.classifier = nn.Sequential(
                                nn.Linear(original_model.fc.in_features, original_model.fc.in_features // 4),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(original_model.fc.in_features // 4, num_classes),
                                nn.Softmax()
                            )
                        elif args.use_dropout == 3:
                            print("[model] no user_feature 2 fc_num with both dropout in resnet")
                            self.classifier = nn.Sequential(
                                nn.Dropout(0.5),
                                nn.Linear(original_model.fc.in_features, original_model.fc.in_features // 4),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(original_model.fc.in_features // 4, num_classes),
                                nn.Softmax()
                            )
                        else:
                            print("[model] please use proper dropout schema")
                            raise AssertionError()
                    else:
                        print("[model] no user_feature 2 fc_num with no dropout in resnet")
                        self.classifier = nn.Sequential(
                            nn.Linear(original_model.fc.in_features, original_model.fc.in_features // 4),
                            nn.ReLU(),
                            nn.Linear(original_model.fc.in_features // 4, num_classes),
                            nn.Softmax()
                        )

                elif args.data_type == 'cu':
                    if args.use_dropout:
                        if args.use_dropout == 1:
                            self.classifier = nn.Sequential(
                                nn.Dropout(0.5),
                                nn.Linear(original_model.fc.in_features + 10 + 1, original_model.fc.in_features // 4),
                                nn.ReLU(),
                                nn.Linear(original_model.fc.in_features // 4, num_classes),
                                nn.Softmax()
                            )
                        elif args.use_dropout == 2:
                            self.classifier = nn.Sequential(
                                nn.Linear(original_model.fc.in_features + 10 + 1, original_model.fc.in_features // 4),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(original_model.fc.in_features // 4, num_classes),
                                nn.Softmax()
                            )
                        elif args.use_dropout == 3:
                            self.classifier = nn.Sequential(
                                nn.Dropout(0.5),
                                nn.Linear(original_model.fc.in_features + 10 + 1, original_model.fc.in_features // 4),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(original_model.fc.in_features // 4, num_classes),
                                nn.Softmax()
                            )
                        else:
                            print("[model] please use proper dropout schema")
                            raise AssertionError()
                    else:
                        self.classifier = nn.Sequential(
                            nn.Linear(original_model.fc.in_features + 10 + 1, original_model.fc.in_features // 4),
                            nn.ReLU(),
                            nn.Linear(original_model.fc.in_features // 4, num_classes),
                            nn.Softmax()
                        )
                else:
                    print("[model] not proper data_type argument")
                    raise AssertionError()

            elif args.fc_num == 3:
                print("[model] using 3 fc layers")
                if args.data_type == 'c':
                    if args.use_dropout:
                        self.classifier = nn.Sequential(
                            nn.Dropout(0.5),
                            nn.Linear(original_model.fc.in_features, original_model.fc.in_features // 2),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(original_model.fc.in_features // 2, original_model.fc.in_features // 4),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(original_model.fc.in_features // 4, num_classes),
                            nn.Softmax()
                        )
                    else:
                        self.classifier = nn.Sequential(
                            nn.Linear(original_model.fc.in_features, original_model.fc.in_features // 2),
                            nn.ReLU(),
                            nn.Linear(original_model.fc.in_features // 2, original_model.fc.in_features // 4),
                            nn.ReLU(),
                            nn.Linear(original_model.fc.in_features // 4, num_classes),
                            nn.Softmax()
                        )
                elif args.data_type == 'cu':
                    if args.use_dropout:
                        self.classifier = nn.Sequential(
                            nn.Dropout(0.5),
                            nn.Linear(original_model.fc.in_features + 10 + 1, original_model.fc.in_features // 2),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(original_model.fc.in_features // 2, original_model.fc.in_features // 4),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(original_model.fc.in_features // 4, num_classes),
                            nn.Softmax()
                        )
                    else:
                        self.classifier = nn.Sequential(
                            nn.Linear(original_model.fc.in_features + 10 + 1, original_model.fc.in_features // 2),
                            nn.ReLU(),
                            nn.Linear(original_model.fc.in_features // 2, original_model.fc.in_features // 4),
                            nn.ReLU(),
                            nn.Linear(original_model.fc.in_features // 4, num_classes),
                            nn.Softmax()
                        )
                else:
                    print("[model] not proper data_type argument")
                    raise AssertionError()
            else:
                print("[model] not proper fc_num argument")
                raise AssertionError()

            if args.open_num == 0:  # fix all feature params
                print("[model] fix all feature params !!")
                for p in self.feature1.parameters():
                    p.requires_grad = False
                for p in self.feature2.parameters():
                    p.requires_grad = False
                for p in self.feature3.parameters():
                    p.requires_grad = False
                for p in self.feature4.parameters():
                    p.requires_grad = False

            elif args.open_num == 1:
                print("[model] fix all feature params except last one")
                for p in self.feature1.parameters():
                    p.requires_grad = False
                for p in self.feature2.parameters():
                    p.requires_grad = False
                for p in self.feature3.parameters():
                    p.requires_grad = False
                for p in self.feature4.parameters():
                    p.requires_grad = True

            elif args.open_num == 2:
                print("[model] do not fix last two feature params  ")
                for p in self.feature1.parameters():
                    p.requires_grad = False
                for p in self.feature2.parameters():
                    p.requires_grad = False
                for p in self.feature3.parameters():
                    p.requires_grad = True
                for p in self.feature4.parameters():
                    p.requires_grad = True

            elif args.open_num == 3:
                print("[model] do not fix last three feature params  ")
                for p in self.feature1.parameters():
                    p.requires_grad = False
                for p in self.feature2.parameters():
                    p.requires_grad = True
                for p in self.feature3.parameters():
                    p.requires_grad = True
                for p in self.feature4.parameters():
                    p.requires_grad = True

            elif args.open_num == 4:
                print("[model] do not fix every param ")
                for p in self.feature1.parameters():
                    p.requires_grad = True
                for p in self.feature2.parameters():
                    p.requires_grad = True
                for p in self.feature3.parameters():
                    p.requires_grad = True
                for p in self.feature4.parameters():
                    p.requires_grad = True
            else:
                print("[model] not proper open_num argument")
                raise AssertionError()

            self.modelName = 'resnet'

        # vgg settings

        elif args.arch.startswith('vgg16'):
            print("[model] using vgg network")
            self.features = original_model.features

            if args.data_type == 'c':
                self.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(25088, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, num_classes),
                    nn.Softmax()
                )
            elif args.data_type == 'cu':
                self.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(25088+10+1, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, num_classes),
                    nn.Softmax()
                )
            else:
                raise AssertionError()

            if args.open_num > 0:
                print( "[model] vgg feature layer open")
                for p in self.features.parameters():
                    p.requires_grad = True
            else:
                print("[model] vgg feature layer fixed")
                for p in self.features.parameters():
                    p.requires_grad = False

            self.modelName = 'vgg16'

        elif args.arch.startswith('densenet169'):
            print("[model] using densenet")

            self.features = original_model.features

            if args.data_type == 'c':

                self.classifier = nn.Sequential(
                    nn.Linear(1664, num_classes),
                    nn.Softmax()
                )

            elif args.data_type == 'cu':
                self.classifier = nn.Sequential(
                    nn.Linear(1664+10+1, num_classes),
                    nn.Softmax()
                )
            else:
                raise AssertionError()

            if args.open_num > 0:
                print( "[model] densenet feature layer open")
                for p in self.features.parameters():
                    p.requires_grad = True
            else:
                print("[model] densenet feature layer fixed")
                for p in self.features.parameters():
                    p.requires_grad = False

            self.modelName = 'densenet'



        else:
            print("[model] Finetuning not supported on this architecture yet")
            raise AssertionError()

        if args.method == 'regression':
            aa =  list(self.classifier.children())[:-1]
            #aa.append(nn.ReLU())
            self.classifier = nn.Sequential(*aa)

    def forward(self, x, y = None, z = None):

        if self.modelName == 'vgg16':
            f = self.features(x)
            f = f.view(f.size(0), -1)

            if self.args.data_type == 'cu':
                y = y.view(-1, 1)
                f = torch.cat((f, y, z), 1)

        elif self.modelName == 'densenet':
            f = self.features(x)
            f = F.relu(f, inplace=True)
            f = F.avg_pool2d(f, kernel_size=7, stride=1).view(f.size(0), -1)

            if self.args.data_type == 'cu':
                y = y.view(-1, 1)
                f = torch.cat((f, y, z), 1)

        elif self.modelName == 'resnet':
            f = self.feature1(x)
            #print(f)
            f = self.feature2(f)
            #print(f)
            f = self.feature3(f)
            #print(f)
            f = self.feature4(f)
            #print(f)
            f = self.avg_pool(f)
            #print(f)
            f = f.view(f.size(0), -1)

            if self.args.data_type == 'cu':
                y = y.view(-1, 1)
                f = torch.cat((f, y, z), 1)

        else:
            raise AssertionError()

        y = self.classifier(f)
        return y

def get_model(args):

    original_model = models.__dict__[args.arch](pretrained=args.use_pretrained)

    if not args.use_pretrained:
        print("[model] do not fix all of the params not using pretrained model")
        args.open_num = 4

    model = FineTuneModel(original_model, args=args)

    if not args.use_pretrained:
        grad_params = [{'params': model.parameters(), 'lr': args.lr}]

    elif args.arch.startswith('resnet'):
        param_list = [{'params': model.feature1.parameters()},
                      {'params': model.feature2.parameters()},
                      {'params': model.feature3.parameters()},
                      {'params': model.feature4.parameters()},
                      {'params': model.classifier.parameters(), 'lr': args.lr}]
        grad_params = param_list[-(args.open_num + 1)::]

    #elif args.arch.startswith('vgg16'):
    else:

        param_list = [{'params': model.features.parameters()},
                      {'params': model.classifier.parameters(), 'lr': args.lr}]

        if args.open_num > 0:
            grad_params = param_list
        else:
            grad_params = param_list[-1]
    '''
    elif args.arch.startswith('densenet'):

        param_list = [{'params': model.features.parameters()},
                      {'params': model.classifier.parameters(), 'lr': args.lr}]

        if args.open_num > 0:
            grad_params = param_list
        else:
            grad_params = param_list[-1]
    else:
        raise AssertionError()
    '''
    if args.use_pretrained: #lr of cnn part will be 0.1*classifier_lr
        lr = args.lr * 0.1
    else:
        lr = args.lr

    ##Optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(grad_params, lr=lr, betas=(0.5, 0.999))
    elif args.optimizer == 'adam_wd':
        optimizer = torch.optim.Adam(grad_params, lr=lr, betas=(0.5, 0.999),weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(grad_params, lr=lr, momentum=0.9, nesterov=True)
    elif args.optimizer == 'sgd_wd':
        optimizer = torch.optim.SGD(grad_params, lr=lr, momentum=0.9, nesterov=True,weight_decay=args.weight_decay)
    else:
        raise AssertionError()

    ##Scheduler
    if args.lr_sch == 0:
        print('scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    elif args.lr_sch == 1:
        print('scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1, last_epoch=-1)')
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1, last_epoch=-1)
    elif args.lr_sch == 2:
        print('scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1, last_epoch=-1)')
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1, last_epoch=-1)
    elif args.lr_sch == 3:
        print('scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones={}, gamma=0.1, last_epoch=-1)'.format(args.milestones))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1, last_epoch=-1)
    elif args.lr_sch == 4:
        print('scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)')
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
    elif args.lr_sch == 5:
        print('scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max,=400 eta_min=0, last_epoch=-1)')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=0, last_epoch=-1)
    elif args.lr_sch == 6:
        print(torch.__version__)
        print('scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=min, factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode=rel, cooldown=0, min_lr=0, eps=1e-08)')

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    else:
        raise AssertionError()

    return model, optimizer, scheduler
