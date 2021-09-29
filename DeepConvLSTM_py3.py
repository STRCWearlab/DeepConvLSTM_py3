import argparse

import os
import time
import random
import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim

import sklearn.metrics as metrics

from datetime import timedelta

from torch.utils.data import DataLoader
from utils import init_weights, makedir, paint, AverageMeter
from datasets import SensorDataset

train_on_gpu = torch.cuda.is_available()  # Check for cuda


## Define our DeepConvLSTM class, subclassing nn.Module.
class DeepConvLSTM(nn.Module):

    def __init__(self, n_channels, n_classes, dataset, experiment='default', conv_kernels=64, kernel_size=5,
                 LSTM_units=128, model='DeepConvLSTM'):
        super(DeepConvLSTM, self).__init__()

        self.conv1 = nn.Conv2d(1, conv_kernels, (kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(n_channels * conv_kernels, LSTM_units, num_layers=2)

        self.classifier = nn.Linear(LSTM_units, n_classes)

        self.activation = nn.ReLU()

        self.model = model
        self.dataset = dataset
        self.experiment = experiment

        makedir(self.path_checkpoints)
        makedir(self.path_logs)
        makedir(self.path_visuals)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.permute(2, 0, 3, 1)

        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)

        x, h = self.lstm(x)
        x = x[-1, :, :]

        out = self.classifier(x)

        return None, out

    @property
    def path_checkpoints(self):
        return f"./models/{self.model}/{self.dataset}/{self.experiment}/checkpoints/"

    @property
    def path_logs(self):
        return f"./models/{self.model}/{self.dataset}/{self.experiment}/logs/"

    @property
    def path_visuals(self):
        return f"./models/{self.model}/{self.dataset}/{self.experiment}/visuals/"


def model_train(model, dataset, dataset_val, args, verbose=False):
    """
    Train model for a number of epochs.

    :param model: A pytorch model
    :param dataset: A SensorDataset containing the data to be used for training the model.
    :param dataset_val: A SensorDataset containing the data to be used for validation of the model.
    :param args: A dict containing config options for the training.
    Required keys:
                    'batch_size': int, number of windows to process in each batch (default 256)
                    'optimizer': str, optimizer function to use. Options: 'Adam' or 'RMSProp'. Default 'Adam'.
                    'lr': float, maximum initial learning rate. Default 0.001.
                    'lr_step': int, interval at which to decrease the learning rate. Default 10.
                    'lr_decay': float, factor by which to  decay the learning rate. Default 0.9.
                    'init_weights': str, How to initialize weights. Options 'orthogonal' or None. Default 'orthogonal'.
                    'epochs': int, Total number of epochs to train the model for. Default 300.
                    'print_freq': int, How often to print loss during each epoch if verbose=True. Default 100.

    :param verbose:
    :return:
    """
    if verbose:
        print(paint("Running HAR training loop ..."))

    loader = DataLoader(dataset, args['batch_size'], True, pin_memory=True)
    loader_val = DataLoader(dataset_val, args['batch_size'], True, pin_memory=True)

    criterion = nn.CrossEntropyLoss(reduction="mean").cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())

    if args['optimizer'] == "Adam":
        optimizer = optim.Adam(params, lr=args['lr'])
    elif args['optimizer'] == "RMSprop":
        optimizer = optim.RMSprop(params, lr=args['lr'])

    if args['lr_step'] > 0:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args['lr_step'], gamma=args['lr_decay']
        )

    if args['init_weights'] == "orthogonal":
        if verbose:
            print(paint("[-] Initializing weights (orthogonal)..."))
        model.apply(init_weights)

    metric_best = 0.0
    start_time = time.time()

    n_epochs = args['epochs']

    for epoch in range(n_epochs):
        if verbose:
            print("--" * 50)
            print("[-] Learning rate: ", optimizer.param_groups[0]["lr"])
        train_one_epoch(model, loader, criterion, optimizer, args, verbose)
        loss, acc, fm, fw = eval_one_epoch(
            model, loader, criterion, args
        )
        start_inf = time.time()
        loss_val, acc_val, fm_val, fw_val = eval_one_epoch(
            model, loader_val, criterion, args
        )
        inf_time = round(time.time() - start_inf)

        if verbose:
            print(
                paint(
                    f"[-] Epoch {epoch}/{args['epochs']}"
                    f"\tTrain loss: {loss:.2f} \tacc: {100 * acc:.2f}(%)\tfm: {100 * fm:.2f}(%)\tfw: {100 * fw:.2f}"
                    f"(%)\tinf:{inf_time}"
                )
            )

            print(
                paint(
                    f"[-] Epoch {epoch}/{args['epochs']}"
                    f"\tVal loss: {loss_val:.2f} \tacc: {100 * acc_val:.2f}(%)\tfm: {100 * fm_val:.2f}(%)"
                    f"\tfw: {100 * fw_val:.2f}(%)\tinf:{inf_time}"
                )
            )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "random_rnd_state": random.getstate(),
            "numpy_rnd_state": np.random.get_state(),
            "torch_rnd_state": torch.get_rng_state(),
        }

        metric = fm_val
        if metric >= metric_best:
            if verbose:
                print(paint(f"[*] Saving checkpoint... ({metric_best}->{metric})", "blue"))
            metric_best = metric
            torch.save(
                checkpoint, os.path.join(model.path_checkpoints, "checkpoint_best.pth")
            )

        if epoch % 5 == 0:
            torch.save(
                checkpoint,
                os.path.join(model.path_checkpoints, f"checkpoint_{epoch}.pth"),
            )

        if args['lr_step'] > 0:
            scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(timedelta(seconds=elapsed))
    if verbose:
        print(paint(f"Finished HAR training loop (h:m:s): {elapsed}"))
        print(paint("--" * 50, "blue"))


def train_one_epoch(model, loader, criterion, optimizer, args, verbose=False):
    losses = AverageMeter("Loss")
    model.train()

    for batch_idx, (data, target, idx) in enumerate(loader):

        data = data.cuda()
        target = target.view(-1).cuda()

        z, logits = model(data)

        loss = criterion(logits, target)

        losses.update(loss.item(), data.shape[0])

        optimizer.zero_grad()

        optimizer.step()

        if verbose:
            if batch_idx % args['print_freq'] == 0:
                print(f"[-] Batch {batch_idx}/{len(loader)}\t Loss: {str(losses)}")


def eval_one_epoch(model, loader, criterion, args):
    losses = AverageMeter("Loss")
    y_true, y_pred = [], []
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target, idx) in enumerate(loader):
            data = data.cuda()
            target = target.cuda()

            z, logits = model(data)
            loss = criterion(logits, target.view(-1))
            losses.update(loss.item(), data.shape[0])

            probabilities = nn.Softmax(dim=1)(logits)
            _, predictions = torch.max(probabilities, 1)

            y_pred.append(predictions.cpu().numpy().reshape(-1))
            y_true.append(target.cpu().numpy().reshape(-1))

    # append invalid samples at the beginning of the test sequence
    if loader.dataset.prefix == "test":
        ws = data.shape[1] - 1
        samples_invalid = [y_true[0][0]] * ws
        y_true.append(samples_invalid)
        y_pred.append(samples_invalid)

    y_true = np.concatenate(y_true, 0)
    y_pred = np.concatenate(y_pred, 0)

    acc = metrics.accuracy_score(y_true, y_pred)
    fm = metrics.f1_score(y_true, y_pred, average="macro")
    fw = metrics.f1_score(y_true, y_pred, average="weighted")

    return losses.avg, acc, fm, fw


def model_eval(model, dataset_test, args, return_results):
    print(paint("Running HAR evaluation loop ..."))

    loader_test = DataLoader(dataset_test, args['batch_size'], False, pin_memory=True)

    criterion = nn.CrossEntropyLoss(reduction="mean").cuda()

    print("[-] Loading checkpoint ...")

    path_checkpoint = os.path.join(model.path_checkpoints, "checkpoint_best.pth")

    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion.load_state_dict(checkpoint["criterion_state_dict"])

    start_time = time.time()

    loss_test, acc_test, fm_test, fw_test = eval_one_epoch(
        model, loader_test, criterion, args
    )

    print(
        paint(
            f"[-] Test loss: {loss_test:.2f}"
            f"\tacc: {100 * acc_test:.2f}(%)\tfm: {100 * fm_test:.2f}(%)\tfw: {100 * fw_test:.2f}(%)"
        )
    )

    elapsed = round(time.time() - start_time)
    elapsed = str(timedelta(seconds=elapsed))
    print(paint(f"[Finished HAR evaluation loop (h:m:s): {elapsed}"))

    if return_results:
        return acc_test, fm_test, fw_test, elapsed


# Command line parser - set hyperparameters based on user input.
def get_args():
    """This function parses and return arguments passed in"""
    parser = argparse.ArgumentParser(
        description='DeepConvLSTM')
    # Add arguments
    parser.add_argument(
        '-d', '--dataset', type=str, help='Target dataset. Default = opportunity.', required=False,
        default='opportunity')
    parser.add_argument(
        '-w', '--window_size', type=int, help='Desired length of sliding window. Default = 24.', required=False,
        default=24)
    parser.add_argument(
        '-s', '--window_step', type=int, help='Desired step size of the sliding window. Default = 12.', required=False,
        default=12)
    parser.add_argument(
        '-e', '--epochs', type=int, help='Number of epochs to train for in total. Default = 300.', required=False,
        default=300)
    parser.add_argument(
        '-lr', '--learning_rate', type=float, help='Initial learning rate. Default = 0.001.', required=False,
        default=0.001)
    parser.add_argument(
        '-ld', '--lr_decay', type=float, help='Learning rate decay factor. Default = 0.1.', required=False,
        default=0.1)
    parser.add_argument(
        '-ls', '--lr_step', type=int, help='Learning rate decay step size. Default = 100.', required=False,
        default=100)

    # Array for all arguments passed to script
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    opp_class_names = ['Null', 'Open Door 1', 'Open Door 2', 'Close Door 1', 'Close Door 2', 'Open Fridge',
                       'Close Fridge', 'Open Dishwasher', 'Close Dishwasher', 'Open Drawer 1', 'Close Drawer 1',
                       'Open Drawer 2', 'Close Drawer 2',
                       'Open Drawer 3', 'Close Drawer 3', 'Clean Table', 'Drink from Cup', 'Toggle Switch']
    config_dataset = {
        "dataset": args.dataset,
        "window": args.window_size,
        "stride": args.window_step,
        "stride_test": 1,
        "path_processed": f"data/{args.dataset}",
    }

    dataset = SensorDataset(**config_dataset, prefix="train", verbose=True)
    dataset_val = SensorDataset(**config_dataset, prefix="val", verbose=True)
    n_channels = dataset.n_channels
    n_classes = dataset.n_classes

    # Define constants
    config_train = {'batch_size': 256,
                    'optimizer': 'Adam',
                    'lr': args.learning_rate,
                    'lr_step': args.lr_step,
                    'lr_decay': args.lr_decay,
                    'init_weights': 'orthogonal',
                    'epochs': args.epochs,
                    'print_freq': 100}

    deepconv = DeepConvLSTM(n_channels=n_channels, n_classes=n_classes, dataset=args.dataset).cuda()

    model_train(deepconv, dataset, dataset_val, config_train, verbose=True)

    dataset_test = SensorDataset(**config_dataset, prefix="test")
    test_config = {'batch_size': 256,
                   'train_mode': False,
                   'dataset': args.dataset}
    model_eval(deepconv, dataset_test, test_config, False)
