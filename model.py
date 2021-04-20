

import matplotlib.pyplot as plt

import torch
import numpy as np

from torch import nn
from torch import optim

from torch.autograd import Variable
import torch.nn.functional as F
import time
from tqdm import tqdm
from utils import *

acc = []
AUC = []
class Encoder(nn.Module):
    """encoder in DA_RNN."""

    def __init__(self, T,
                 input_size,
                 encoder_num_hidden,
                 parallel=False):
        """Initialize an encoder in DA_RNN."""
        super(Encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.parallel = parallel
        self.T = T

        # Fig 1. Temporal Attention Mechanism: Encoder is LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.encoder_num_hidden,
            num_layers=1
        )

        # Construct Input Attention Mechanism via deterministic attention model
        # Eq. 8: W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        self.encoder_attn = nn.Linear(
            in_features=2 * self.encoder_num_hidden + self.T - 1,
            out_features=1
        )
        #self.encoder_sigmoid = 1.0 / (1 + np.exp(-(self.encoder_value)))

    def forward(self, X):
        """forward.

        Args:
            X: input data

        """
        X_tilde = Variable(X.data.new(
            X.size(0), self.T - 1, self.input_size).zero_())                    #torch.Size([128, 9, 81])
        ##print(X_tilde.shape)
        X_encoded = Variable(X.data.new(
            X.size(0), self.T - 1, self.encoder_num_hidden).zero_())             #torch.Size([128, 9, 128])
        ##print(X_encoded.shape)
        # Eq. 8, parameters not in nn.Linear but to be learnt
        # v_e = torch.nn.Parameter(data=torch.empty(
        #     self.input_size, self.T).uniform_(0, 1), requires_grad=True)
        # U_e = torch.nn.Parameter(data=torch.empty(
        #     self.T, self.T).uniform_(0, 1), requires_grad=True)

        # h_n, s_n: initial states with dimention hidden_size
        h_n = self._init_states(X)                                              #torch.Size([1, 128, 128])
        s_n = self._init_states(X)                                              #torch.Size([1, 128, 128])
        for t in range(self.T - 1):
            # batch_size * input_size * (2 * hidden_size + T - 1)
            #permute是交换矩阵维度
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim=2)                          #torch.Size([128, 81, 265])
            ##print('1',x.shape)
            x = self.encoder_attn(
                x.view(-1, self.encoder_num_hidden * 2 + self.T - 1))           #torch.Size([10368, 1])
            ##print('2', x.shape)

            ####这里加一个sigmoid
            x = torch.tanh(x)

            # get weights by softmax
            alpha = F.sigmoid(x.view(-1, self.input_size))    #这个地方是权重矩阵可以改进  原本为softmax
            #print(alpha)
            # get new input for LSTM
            x_tilde = torch.mul(alpha, X[:, t, :])                          #torch.Size([128, 81])
            ##print('x_tilde',x_tilde.shape)

            # Fix the warning about non-contiguous memory
            # https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.encoder_lstm.flatten_parameters()

            # encoder LSTM
            _, final_state = self.encoder_lstm(
                x_tilde.unsqueeze(0), (h_n, s_n))
            h_n = final_state[0]                                            #torch.Size([1, 128, 128])
            s_n = final_state[1]                                            #torch.Size([1, 128, 128])
            ##print('h_n', h_n.shape)
            ##print('s_n', s_n.shape)


            X_tilde[:, t, :] = x_tilde
            X_encoded[:, t, :] = h_n
            ##print('X_tilde', X_tilde.shape)                                 #torch.Size([128, 9, 81])
            ##print('X_encoded', X_encoded.shape)                             #torch.Size([128, 9, 81])
        return X_tilde, X_encoded

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder."""
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(1, X.size(0), self.encoder_num_hidden).zero_())


class Decoder(nn.Module):
    """decoder in DA_RNN."""

    def __init__(self, T, decoder_num_hidden, encoder_num_hidden):
        """Initialize a decoder in DA_RNN."""
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.encoder_num_hidden = encoder_num_hidden
        self.T = T

        self.attn_layer = nn.Sequential(
            nn.Linear(2 * decoder_num_hidden +
                      encoder_num_hidden, encoder_num_hidden),
            nn.Tanh(),
            nn.Linear(encoder_num_hidden, 1),
            #nn.Sigmoid()
        )
        self.lstm_layer = nn.LSTM(
            input_size=1,
            hidden_size=decoder_num_hidden
        )
        self.fc = nn.Linear(encoder_num_hidden + 1, 1)
        self.fc_final = nn.Linear(decoder_num_hidden + encoder_num_hidden, 1)

        self.fc.weight.data.normal_()

    def forward(self, X_encoded, y_prev):
        """forward."""
        d_n = self._init_states(X_encoded)                          #torch.Size([1, 128, 128])
        ##print('d_n', d_n.shape)
        c_n = self._init_states(X_encoded)                          #torch.Size([1, 128, 128])
        ##print('c_n', c_n.shape)

        for t in range(self.T - 1):
            #print('X_encoded', X_encoded.shape)
            x = torch.cat((d_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           c_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           X_encoded), dim=2)                       #torch.Size([128, 9, 384])
            ##print('x', x.shape)
            beta = self.attn_layer(x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T - 1)
            beta = beta.view(self.T-1,-1)
            #print(beta)
            #beta = torch.mean(self.attn_layer(x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T - 1))  #torch.Size([128, 9])
            #print('beta',F.softmax(beta)[:,:30])
            beta = F.softmax(beta)
            beta = beta.view(-1, self.T-1)
            # Eqn. 14: compute context vector
            # batch_size * encoder_hidden_size
            context = torch.bmm(beta.unsqueeze(1),X_encoded)[:, 0, :]              #torch.Size([128, 128])
            ##print('context', context.shape)
            if t < self.T - 1:
                # Eqn. 15
                # batch_size * 1
                #print('y_prev[:, t].unsqueeze(1)', y_prev[:, t].unsqueeze(1).shape)
                y_tilde = self.fc(torch.cat((context,y_prev[:, t].unsqueeze(1)),dim = 1))
                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, final_states = self.lstm_layer(
                    y_tilde.unsqueeze(0), (d_n, c_n))

                d_n = final_states[0]  # 1 * batch_size * decoder_num_hidden
                c_n = final_states[1]  # 1 * batch_size * decoder_num_hidden

        # Eqn. 22: final output
        #print(d_n[0])
        y_pred = self.fc_final(torch.cat((d_n[0], context), dim=1))
        ''''
        #print(y_pred)
        y_pred = y_pred.cpu()
        #y_pred = torch.max(y_pred, dim=1)[1].data.numpy()
        #y_pred = torch.tensor(y_pred,dtype=torch.float32)
        y_pred_n = []
        y_pred_temp =y_pred.detach().numpy()
        y_pred_avg = np.mean(y_pred_temp)
        #print(y_pred_avg)
        for i in range(len(y_pred)):
            if y_pred[i] > y_pred_avg:
                y_pred_n.append(1)
            elif y_pred[i] <= y_pred_avg:
                y_pred_n.append(0)
        y_pred = torch.tensor(y_pred_n)
        y_pred = y_pred.cuda()
        #print(y_pred)
        '''
        return y_pred

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder."""
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(1, X.size(0), self.decoder_num_hidden).zero_())


class DA_RNN(nn.Module):
    """Dual-Stage Attention-Based Recurrent Neural Network."""

    def __init__(self, X, y, T,
                 encoder_num_hidden,
                 decoder_num_hidden,
                 batch_size,
                 learning_rate,
                 epochs,
                 parallel=False):
        """initialization."""
        super(DA_RNN, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.parallel = parallel
        self.shuffle = False
        self.epochs = epochs
        self.T = T
        self.X = X
        self.y = y

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)

        self.Encoder = Encoder(input_size=X.shape[1],
                               encoder_num_hidden=encoder_num_hidden,
                               T=T).to(self.device)
        self.Decoder = Decoder(encoder_num_hidden=encoder_num_hidden,
                               decoder_num_hidden=decoder_num_hidden,
                               T=T).to(self.device)

        # Loss function
        self.criterion = nn.MSELoss()

        if self.parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Encoder.parameters()),
                                            lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Decoder.parameters()),
                                            lr=self.learning_rate)

        # Training set
        self.train_timesteps = int(self.X.shape[0] * 0.7)
        #self.y = self.y - np.mean(self.y[:self.train_timesteps])
        self.y = self.y                            #此处输入不要减平均值直接0  1
        self.input_size = self.X.shape[1]

    def train(self):
        """Training process."""
        iter_per_epoch = int(
            np.ceil(self.train_timesteps * 1. / self.batch_size))
        self.iter_losses = np.zeros(self.epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(self.epochs)

        n_iter = 0

        for epoch in tqdm(range(self.epochs)):
            since = time.time()
            if self.shuffle:
                ref_idx = np.random.permutation(self.train_timesteps - self.T)
            else:
                ref_idx = np.array(range(self.train_timesteps - self.T))

            idx = 0

            while (idx < self.train_timesteps):
                # get the indices of X_train
                indices = ref_idx[idx:(idx + self.batch_size)]
                # x = np.zeros((self.T - 1, len(indices), self.input_size))
                x = np.zeros((len(indices), self.T - 1, self.input_size))
                y_prev = np.zeros((len(indices), self.T - 1))
                y_gt = self.y[indices + self.T]

                # format x into 3D tensor
                for bs in range(len(indices)):
                    x[bs, :, :] = self.X[indices[bs]:(
                        indices[bs] + self.T - 1), :]
                    y_prev[bs, :] = self.y[indices[bs]: (indices[bs] + self.T - 1)]

                loss = self.train_forward(x, y_prev, y_gt)
                self.iter_losses[int(
                    epoch * iter_per_epoch + idx / self.batch_size)] = loss

                idx += self.batch_size
                n_iter += 1

                if n_iter % 10000 == 0 and n_iter != 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9

                self.epoch_losses[epoch] = np.mean(self.iter_losses[range(
                    epoch * iter_per_epoch, (epoch + 1) * iter_per_epoch)])

            if epoch % 1000 == 0:
                print("Epochs: ", epoch, " Iterations: ", n_iter,
                      " Loss: ", self.epoch_losses[epoch])

            if epoch % 1000 == 0:
                y_train_pred = self.test(on_train=True)
                y_test_pred = self.test(on_train=False)
                y_pred = np.concatenate((y_train_pred, y_test_pred))
                plt.ioff()
                plt.figure(epoch)
                #plt.plot(range(1, 1 + len(self.y)), self.y, label="True")
                plt.ylim(0, 1.1)
                plt.scatter(range(self.T, len(y_train_pred) + self.T),
                         y_train_pred,1, label='Predicted - Train')
                plt.scatter(range(self.T + len(y_train_pred), len(self.y) + 1),
                         y_test_pred,1, label='Predicted - Test')
                plt.legend(loc='upper left')

                plt.savefig('第'+str(epoch)+".png")


                p = 0.0
                y_pred_n = []
                # P=0
                # N=0
                # y_train_pred_min = np.min(y_train_pred)
                # for i in range(len(y_train_pred)):
                #     if y_train_pred[i]>y_train_pred_min+0.1:
                #         y_pred_n.append(1)
                #         P+=1
                #     elif y_train_pred[i]<=y_train_pred_min+0.1:
                #         y_pred_n.append(0)
                #         N+=1
                # print(set(y_pred_n))
                # print(N,P)
                # for i in range(len(y_pred_n)):
                #     if self.y[i] == y_pred_n[i]:
                #         p += 1
                # AUC.append(calcAUC_byRocArea(self.y[:len(y_pred_n)],y_pred_n))
                # acc.append(p / len(y_train_pred))
                #print(acc)
                #plt.show()

                #偏置
                y_test_pred_min = np.min(y_test_pred)
                for i in range(len(y_test_pred)):
                    if y_test_pred[i] > y_test_pred_min + 0.1:
                        y_pred_n.append(1)
                    elif y_test_pred[i] <= y_test_pred_min + 0.1:
                        y_pred_n.append(0)
                for i in range(len(y_pred_n)):
                    if self.y[len(y_train_pred)+i] == y_pred_n[i]:
                        p += 1
                acc.append(p / len(y_test_pred))
                AUC.append(calcAUC_byRocArea(self.y[len(y_train_pred):], y_pred_n))
                data_write_csv('预测结果.csv', y_pred_n)


            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


        '''
                #for i in range(len(y_train_pred)):
                y_pred_m = self.sigmoid(y_pred)
                y_m = self.sigmoid(self.y)
                y_pred_n = []
                for i in range(len(y_pred_m)):
                    if abs(y_pred_m[i]-y_m.max())>abs(y_pred_m[i]-y_m.min()):
                        y_pred_n.append(1)
                    elif abs(y_pred_m[i] - y_m.max()) <= abs(y_pred_m[i] - y_m.min()):
                        y_pred_n.append(0)
                y_tranc=[]
                for i in range(len(y_m)):
                    if y_m[i] == y_m.max():
                        y_tranc.append(1)
                    elif y_m[i] == y_m.min():
                        y_tranc.append(0)
                p = 0.0
                for i in range(len(y_pred_n)):
                    if y_tranc[i] == y_pred_n[i]:
                        p += 1
                acc = []
                acc.append(p / len(y_pred_m))
                print(acc)

                #acc =
                #plt.plot(acc)
                plt.legend(loc='upper left')
                '''

    '''
                    p=0
                    acc = []
                    pl=np.array(y_pred)
                    pltrans=np.array([(i-pl.min())/(pl.max()-pl.min()) for i in pl ])
                    y_tranc=[]
                    re_trans=[]
                    y_mid = (self.y.max()-self.y.min())/2
                    print(y_mid)
                    for i in self.y:
                        if i >y_mid:
                            y_tranc.append(1)
                        elif i <= y_mid:
                            y_tranc.append(0)
                    for i in pltrans:
                        if i >=0.5:
                            re_trans.append(1)
                        else:
                            re_trans.append(0)
                    for i in range(len(re_trans)):
                        if re_trans[i] == y_tranc[i]:
                            p+=1

                    acc.append(p/len(y_pred))
                    print(acc)
    '''

    def train_forward(self, X, y_prev, y_gt):
        """Forward pass."""
        # zero gradients                                                                    #X.shape  (128, 9, 81)
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_weighted, input_encoded = self.Encoder(
            Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device)))          #torch.Size([128, 9, 81])  ([128, 9, 128])
        #print('X',X.shape)
        #print('input_weighted', input_weighted.shape)
        #print('input_encoded', input_encoded.shape)
        y_pred = self.Decoder(input_encoded, Variable(
            torch.from_numpy(y_prev).type(torch.FloatTensor).to(self.device)))              #torch.Size([128, 1])
        #print('y_pred', y_pred.shape)
        y_true = Variable(torch.from_numpy(
            y_gt).type(torch.FloatTensor).to(self.device))

        y_true = y_true.view(-1, 1)
        loss = self.criterion(y_pred, y_true)
        #print(loss)
        loss.requires_grad_(True)
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()

    def test(self, on_train=False):
        """Prediction."""

        if on_train:
            y_pred = np.zeros(self.train_timesteps - self.T + 1)
        else:
            y_pred = np.zeros(self.X.shape[0] - self.train_timesteps)

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i: (i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T - 1, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), self.T - 1))

            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X[range(
                        batch_idx[j], batch_idx[j] + self.T - 1), :]
                    y_history[j, :] = self.y[range(
                        batch_idx[j], batch_idx[j] + self.T - 1)]
                else:
                    X[j, :, :] = self.X[range(
                        batch_idx[j] + self.train_timesteps - self.T, batch_idx[j] + self.train_timesteps - 1), :]
                    y_history[j, :] = self.y[range(
                        batch_idx[j] + self.train_timesteps - self.T, batch_idx[j] + self.train_timesteps - 1)]

            y_history = Variable(torch.from_numpy(
                y_history).type(torch.FloatTensor).to(self.device))
            _, input_encoded = self.Encoder(
                Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device)))

            y_pred[i:(i + self.batch_size)] = self.Decoder(input_encoded,
                                                           y_history).cpu().data.numpy()[:,0]
            i += self.batch_size

        return y_pred
