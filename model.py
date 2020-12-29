import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self, input_ch, output_ch, k=3):
        super(CNN, self).__init__()
        self.CNN_seq = nn.Sequential(
            nn.Conv1d(input_ch, output_ch, kernel_size=k, padding=(k // 2)),
            nn.BatchNorm1d(output_ch), nn.PReLU(), nn.MaxPool1d(2))

    def forward(self, x):
        return self.CNN_seq(x)

class Q_vals(nn.Module):
    def __init__(self, input_ft, n_class=2):
        super(Q_vals, self).__init__()
        self.Q_vals_seq = nn.Sequential(nn.Linear(input_ft, 1024), nn.PReLU(),
                                        nn.Linear(1024, 1024), nn.PReLU(),
                                        nn.Linear(1024, n_class))

    def forward(self, x):
        return self.Q_vals_seq(x)

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, length, chkpt_path, input_ch=2):
        super(DuelingDeepQNetwork, self).__init__()
        self.c1 = CNN(input_ch, input_ch * 2)
        self.c2 = CNN(input_ch * 2, input_ch * 4)
        self.c3 = CNN(input_ch * 4, input_ch * 8)
        self.c4 = CNN(input_ch * 8, input_ch * 8)

        self.V = Q_vals(length // 16 * 8 * input_ch, 1)
        self.A = Q_vals(length // 16 * 8 * input_ch, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:1' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.chkpt_path = chkpt_path

        
    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)

        _x_ = T.flatten(x, start_dim=1)

        V = self.V(_x_)
        A = self.A(_x_)
        
        return V, A

    def save_checkpoint(self, i):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), os.path.join(self.chkpt_path, f"best_weights{i}.pth"))

    def load_checkpoint(self, i):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(os.path.join(self.chkpt_path, f"best_weights{i}.pth"), map_location=self.device))

