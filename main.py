

import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.autograd import Variable

from utils import *
from model import *

import warnings

warnings.filterwarnings("ignore")

def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="PyTorch implementation of paper 'A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction'")

    # Dataset setting
    parser.add_argument('--dataroot', type=str, default="10万.csv", help='path to dataset')
    parser.add_argument('--batchsize', type=int, default=10000, help='input batch size [128]')

    # Encoder / Decoder parameters setting
    parser.add_argument('--nhidden_encoder', type=int, default=128, help='size of hidden states for the encoder m [64, 128]')
    parser.add_argument('--nhidden_decoder', type=int, default=128, help='size of hidden states for the decoder p [64, 128]')
    parser.add_argument('--ntimestep', type=int, default=4, help='the number of time steps in the window T [10]')

    # Training parameters setting
    parser.add_argument('--epochs', type=int, default=100000, help='number of epochs to train [10, 200, 500]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate [0.001] reduced by 0.1 after each 10000 iterations')

    # parse the arguments
    args = parser.parse_args()

    return args


def main():
    """Main pipeline of DA-RNN."""
    args = parse_args()

    # Read dataset
    print("==> Load dataset ...")
    X, y = read_data(args.dataroot, debug=False)
    X = X[:,1:]
    print(X.shape)
    print(y.shape)

    # Initialize model
    print("==> Initialize DA-RNN model ...")
    model = DA_RNN(
        X,
        y,
        args.ntimestep,
        args.nhidden_encoder,
        args.nhidden_decoder,
        args.batchsize,
        args.lr,
        args.epochs
    )

    # Train
    print("==> Start training ...")
    model.train()
    # Prediction
    y_pred = model.test()



    #plt.ion()
    fig1 = plt.figure()
    plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
    #plt.pause(0.1)
    #plt.ioff()
    plt.savefig("1.png")
    plt.close(fig1)

    fig2 = plt.figure()
    plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
    plt.savefig("2.png")
    plt.close(fig2)

    # fig3 = plt.figure()
    # plt.plot(model.y[model.train_timesteps:], label="True")
    # plt.plot(y_pred, label='Predicted')
    # plt.legend(loc='upper left')
    # plt.savefig("3.png")
    # plt.close(fig3)
    # print('Finished Training')
    fig3 = plt.figure()
    print(acc)
    plt.plot(acc, label="ACC")
    plt.legend(loc='upper left')
    plt.savefig("3.png")
    plt.close(fig3)


    fig4 = plt.figure()
    print('AUC',AUC)
    plt.plot(AUC)
    plt.savefig("4.png")
    plt.close(fig4)
    print('Finished Training')

if __name__ == '__main__':
    main()
