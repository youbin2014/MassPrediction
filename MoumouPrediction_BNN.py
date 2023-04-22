import pandas as pd
import numpy as np
from simple_FCN import FCN
# from simple_FCN import FCN_Det
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import BatchSampler


def train(args, model, device, X_train,y_train, optimizer, epoch):
    model.train()
    batch_size=20
    sampler = BatchSampler(range(len(X_train)), batch_size=batch_size, drop_last=False)
    batch_idx=0
    for batch_indices in sampler:
        data = X_train[batch_indices]
        target = y_train[batch_indices]
        data, target = torch.tensor(data).double().to(device), torch.tensor(target).double().to(device)
        optimizer.zero_grad()
        output_ = []
        kl_ = []
        for mc_run in range(args.num_mc):
            # output, kl = model(data)
            output = model(data)
            kl=torch.tensor([0]).double().cuda()
            output_.append(output)
            kl_.append(kl)
        output = torch.mean(torch.stack(output_), dim=0)
        kl = torch.mean(torch.stack(kl_), dim=0)
        Huber_loss = F.smooth_l1_loss(output[:,0], target)
        #ELBO loss
        loss = Huber_loss + (kl / args.batch_size)

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(X_train),
                100. * batch_idx / len(X_train), loss.item()))
        batch_idx+=1


def test(args, model, device, X_test,y_test, epoch):
    model.eval()
    test_loss = 0
    abs_loss=0
    correct = 0
    with torch.no_grad():
        batch_size = 1
        sampler = BatchSampler(range(len(X_test)), batch_size=batch_size, drop_last=False)

        for batch_indices in sampler:
            data = X_test[batch_indices]
            target = y_test[batch_indices]
            data, target = torch.tensor(data).double().to(device), torch.tensor(target).double().to(device)
            output, kl = model(data)
            # output = model(data)
            # kl=torch.tensor([0]).double().cuda()
            test_loss += F.smooth_l1_loss(output[:,0], target, reduction='sum').item()+ (kl / args.batch_size)  # sum up batch loss

            abs_loss += np.abs(scaler.inverse_transform(np.array([output[:,0].item()]).reshape(-1, 1)).flatten()- scaler.inverse_transform(np.array([target.item()]).reshape(-1, 1)).flatten())

    test_loss /= len(X_test)
    abs_loss /= len(X_test)

    print(
        '\nTest set: Average loss: {:.4f}, abs loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss,abs_loss[0], correct, len(X_test),
            100. * correct / len(X_test)))

    val_accuracy = correct / len(X_test)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=10000,
                        metavar='N',
                        help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs',
                        type=int,
                        default=500,
                        metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.1,
                        metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.7,
                        metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./checkpoint/bayesian')
    parser.add_argument('--mode', type=str, required=True, help='train | test')
    parser.add_argument(
        '--num_monte_carlo',
        type=int,
        default=20,
        metavar='N',
        help='number of Monte Carlo samples to be drawn for inference')
    parser.add_argument('--num_mc',
                        type=int,
                        default=1,
                        metavar='N',
                        help='number of Monte Carlo runs during training')

    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs/mnist/bayesian',
        metavar='N',
        help=
        'use tensorboard for logging and visualization of training progress')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    # read the xlsx file into a pandas dataframe
    df = pd.read_excel('./AMEoutputexp.xlsx')

    # convert the dataframe into a numpy array
    data = np.array(df)

    import numpy as np
    from sklearn.model_selection import train_test_split

    # assume X is your numpy array dataset and y is your target variable
    X = np.array(data[:,(1,2,3)]).astype(np.float32)
    y = np.array(data[:,13]).astype(np.float32)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    y = scaler.fit_transform(y.reshape(-1, 1)).flatten()
    scaler2 = MinMaxScaler()
    X = scaler2.fit_transform(X)
    #
    # from matplotlib import pyplot as plt
    #
    # plt.plot(y)
    # plt.show()

    # scaler.inverse_transform(x_normalized.reshape(-1, 1)).flatten()

    # split the dataset into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # print the shapes of the resulting arrays
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    model = FCN()
    # model = FCN_Det()
    model = model.double().cuda()
    device = torch.device("cuda" if use_cuda else "cpu")
    lr = args.lr

    for epoch in range(1, args.epochs + 1):
        if epoch>0 and epoch%200==0:
            lr=lr/2
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        train(args, model, device, X_train,y_train, optimizer, epoch)
        test(args, model, device, X_test,y_test, epoch)
        scheduler.step()

        torch.save(model.state_dict(),
                   args.save_dir + "/moumouBNN.pth")


