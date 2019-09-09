#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fine-tune all layers for bilinear CNN.

This is the second step.
"""

import os
import time

import torch
import torchvision
import torch.nn as nn
from FG_Encoding import cub200 as cub200, ft_model
from FG_Encoding import utils


# mport model as model

# torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.FloatTensor)
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.benchmark = True


class FGTtrain(object):
    """Manager class to train bilinear CNN.

    Attributes:
        _is_all, bool: In the all/fc phase.
        _options, dict<str, float/int>: Hyperparameters.
        _paths, dict<str, str>: Useful paths.
        _net, torch.nn.Module: Bilinear CNN.
        _criterion, torch.nn.Module: Cross-entropy loss.
        _optimizer, torch.optim.Optimizer: SGD with momentum.
        _scheduler, tirch.optim.lr_scheduler: Reduce learning rate when plateau.
        _train_loader, torch.utils.data.DataLoader.
        _test_loader, torch.utils.data.DataLoader.
    """

    def __init__(self, paths, LOAD_MODEL):
        print("PyTorch Version: ", torch.__version__)
        print("Torchvision Version: ", torchvision.__version__)
        """Prepare the network and data.

        Args:
            paths, dict<str, str>: Useful paths.
        """
        print('Prepare the network and data.')

        # Configurations.
        self._paths = paths

        # Network.
        if LOAD_MODEL is not None:
            print('start load!!!!!')
            self._net = ft_model.Model()
            state_dict = torch.load(LOAD_MODEL)
            self._net.load_state_dict(state_dict['state_dict'])
        else:
            #print(None)
            self._net = ft_model.Model()
        # self._net = self._net.cuda()

        print('net loading!----finished!')
        print(self._net)

        resize_size = 256
        crop_size = 224
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((resize_size, resize_size)),
            torchvision.transforms.RandomCrop((crop_size, crop_size)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225)),
        ])
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((resize_size, resize_size)),
            torchvision.transforms.CenterCrop((crop_size, crop_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225)),
        ])

        train_data = cub200.CUB200(
            root=self._paths['cub200'], train=True, transform=train_transforms,
            download=False)
        test_data = cub200.CUB200(
            root=self._paths['cub200'], train=False, transform=test_transforms,
            download=False)

        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=1, shuffle=True, num_workers=4, pin_memory=False)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=8, shuffle=False, num_workers=4, pin_memory=False)

        self._criterion = nn.CrossEntropyLoss()

        self._head = list(map(id, self._net.head.parameters()))
        # self._fc_2 = list(map(id, self._net.fc_2.parameters()))
        # self._fc_1 = list(map(id, self._net.fc_1.parameters()))
        self._no_head = filter(lambda p: id(p) not in self._head,
                               self._net.parameters())

        # self._optimizer = torch.optim.SGD([
        #     {'params': self._no_fc, 'lr': 1e-3},
        #     {'params': self._net.fc_1.parameters(), 'lr': 1e-3 * 10}],
        #     momentum=0.9, weight_decay=1e-4)

        self._optimizer = torch.optim.SGD([
            {'params': self._no_head, 'lr': 1e-3},
            {'params': self._net.head.parameters(), 'lr': 1e-3 * 10}
        ], momentum=0.9, weight_decay=1e-4)

        # print(len(self._optimizer[False].param_groups))
        # self._optimizer_0 = torch.optim.Adam(utils.gather_the_parameters(self._net, feature_extract=True), lr=1e-4,
        #                                      weight_decay=1e-5)
        #self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=30, gamma=0.1, last_epoch=-1)
        # self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self._optimizer, mode='max', factor=0.1, patience=8, verbose=True,
        #     threshold=1e-4)

    def train(self):
        """Train the network."""
        print('Training.')

        self._net.train()
        best_acc = 0.0
        best_epoch = None
        print('Epoch\tTrain loss\tTrain acc\tTest acc\tTime')
        for t in range(100):
            #self._scheduler.step(t)
            epoch_loss = []
            num_correct = 0
            num_total = 0
            tic = time.time()
            batch_num = 0

            if t == 0:
                utils.set_parameter_requires_grad(self._net.head.parameters(), requires_grad=True)
                utils.gather_the_parameters(self._net)

            else:
                utils.set_parameter_requires_grad(self._net.parameters(), requires_grad=True)
                utils.gather_the_parameters(self._net)

            for instances, labels in self._train_loader:
                batch_num = batch_num + 1
                # Data.
                # instances = instances.cuda()
                # labels = labels.cuda()

                # Forward pass.
                fea = self._net(instances)
                loss = self._criterion(fea, labels)

                print('batch: {}        loss:{}'.format(batch_num, loss))

                with torch.no_grad():
                    epoch_loss.append(loss.item())
                    # Prediction.
                    prediction = torch.argmax(fea, dim=1)
                    num_total += labels.size(0)
                    num_correct += torch.sum(prediction == labels).item()

                # Backward pass.
                # if t == 0:
                #     self._optimizer_0.zero_grad()
                #     loss.backward()
                #     self._optimizer_0.step()
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                del instances, labels, fea, loss, prediction
            train_acc = 100 * num_correct / num_total
            test_acc = self._accuracy(self._test_loader)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = t + 1
                print('*', end='')
                save_path = os.path.join(
                    self._paths['model'],
                    'bcnn_%s_epoch_%d.pth' % ('v2', t))
                torch.save(self._net.state_dict(), save_path)
            toc = time.time()
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2f min' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_acc,
                   test_acc, (toc - tic) / 60))
            # self._scheduler.step(test_acc)
        print('Best at epoch %d, test accuaray %4.2f' % (best_epoch, best_acc))

    def test(self):
        test_acc = self._accuracy(self._test_loader)
        print('test accuracy:{}'.format(test_acc))

    def _accuracy(self, data_loader):
        """Compute the train/test accuracy.

        Args:
            data_loader: Train/Test DataLoader.

        Returns:
            Train/Test accuracy in percentage.
        """
        with torch.no_grad():
            self._net.eval()
            num_correct = 0
            num_total = 0
            for instances, labels in data_loader:
                # Data.
                # instances = instances.cuda()
                # labels = labels.cuda()

                instances = instances
                labels = labels

                # Forward pass.
                score = self._net(instances)

                # Predictions.
                prediction = torch.argmax(score, dim=1)
                num_total += labels.size(0)
                num_correct += torch.sum(prediction == labels).item()
            self._net.train()  # Set the model to training phase
        return 100 * num_correct / num_total


def main():
    """The main function."""
    # parser = argparse.ArgumentParser(
    #     description='Train mean field bilinear CNN on CUB200.')
    # parser.add_argument('--base_lr', dest='base_lr', type=float, required=True,
    #                     help='Base learning rate for training.')
    # parser.add_argument('--batch_size', dest='batch_size', type=int,
    #                     required=True, help='Batch size.')
    # parser.add_argument('--epochs', dest='epochs', type=int, required=True,
    #                     help='Epochs for training.')
    # parser.add_argument('--weight_decay', dest='weight_decay', type=float,
    #                     required=True, help='Weight decay.')
    # parser.add_argument('--pretrained', dest='pretrained', type=str,
    #                     required=False, help='Pre-trained model.')
    # args = parser.parse_args()
    # if args.base_lr <= 0:
    #     raise AttributeError('--base_lr parameter must >0.')
    # if args.batch_size <= 0:
    #     raise AttributeError('--batch_size parameter must >0.')
    # if args.epochs < 0:
    #     raise AttributeError('--epochs parameter must >=0.')
    # if args.weight_decay <= 0:
    #     raise AttributeError('--weight_decay parameter must >0.')

    project_root = os.popen('pwd').read().strip()
    # options = {
    #     'base_lr': args.base_lr,
    #     'batch_size': args.batch_size,
    #     'epochs': args.epochs,
    #     'weight_decay': args.weight_decay,
    # }
    # paths = {
    #     'cub200': os.path.join(project_root, 'data', 'cub200'),
    #     'aircraft': os.path.join(project_root, 'data', 'aircraft'),
    #     'model': os.path.join(project_root, 'model'),
    #     'pretrained': (os.path.join(project_root, 'model', args.pretrained)
    #                    if args.pretrained else None),
    # }

    paths = {
        # 'cub200': '/disks/disk0/tpq/datasets/CUB_200_2011/',
        'cub200': '/Users/peggytang/Downloads/CUB_200_2011/',
        'model': '/disks/disk0/tpq/datasets/CUB_200_2011/model/'
    }
    # LOAD_MODEL = '/disks/disk0/tpq/bcnn_224_epoch_132.pth'
    LOAD_MODEL = None

    # for d in paths:
    #     if d == 'pretrained':
    #         assert paths[d] is None or os.path.isfile(paths[d])
    #     else:
    #         assert os.path.isdir(paths[d])

    manager = FGTtrain(paths, LOAD_MODEL)
    if LOAD_MODEL is not None:
        manager.test()
    else:
        manager.train()


if __name__ == '__main__':
    main()
