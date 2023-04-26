from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.layers import Conv2dReparameterization
from bayesian_torch.layers import LinearReparameterization

prior_mu = 0.0
prior_sigma = 0.01
posterior_mu_init = 0.0
posterior_rho_init = -3.0

class FCN_Det(nn.Module):
    def __init__(self,in_c=3):
        super(FCN_Det, self).__init__()

        self.fc1 = nn.Linear(
            in_features=in_c,
            out_features=32
        )
        self.fc2 = nn.Linear(
            in_features=32,
            out_features=32,
        )
        self.fc3 = nn.Linear(
            in_features=32,
            out_features=1
        )
        self.dropout=nn.Dropout(0.25)

        self.Lrelu = nn.LeakyReLU()
    def forward(self, x):
        x= self.fc1(x)
        x = self.Lrelu(x)
        x = self.fc2(x)
        x = self.Lrelu(x)
        # x=self.dropout(x)
        x = self.fc3(x)
        output=x
        return output


class FCN(nn.Module):
    def __init__(self,in_c=3):
        super(FCN, self).__init__()

        self.fc1 = LinearReparameterization(
            in_features=in_c,
            out_features=32,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )
        self.fc2 = LinearReparameterization(
            in_features=32,
            out_features=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )


    def forward(self, x):
        kl_sum = 0

        x, kl = self.fc1(x)
        kl_sum += kl
        x = F.relu(x)
        x, kl = self.fc2(x)
        kl_sum += kl
        output=x
        return output, kl_sum


import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_mixtures):
        super(Net, self).__init__()
        self.num_mixtures = num_mixtures
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_mixtures * 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MixtureDensityNetwork(nn.Module):
    def __init__(self, num_mixtures):
        super(MixtureDensityNetwork, self).__init__()
        self.num_mixtures = num_mixtures
        self.net = Net(num_mixtures)

    def forward(self, x):
        x = self.net(x)

        # Split output into mixture weights, means, and variances
        mixture_weights = F.softmax(x[:, :self.num_mixtures], dim=1)
        means = x[:, self.num_mixtures:self.num_mixtures * 2]
        variances = F.softplus(x[:, self.num_mixtures * 2:])

        return mixture_weights, means, variances

    def loss(self, x, y):
        # Calculate the negative log-likelihood of the data
        mixture_weights, means, variances = self.forward(x)
        diff = torch.unsqueeze(y, dim=1) - means
        exponent = -0.5 * torch.div(torch.pow(diff, 2), variances)
        normalization = torch.sqrt(2 * torch.tensor([3.1415926])) * torch.sqrt(variances)
        pdf = torch.div(torch.exp(exponent), normalization)
        weighted_pdf = torch.mul(pdf, mixture_weights)
        sum_pdf = torch.sum(weighted_pdf, dim=1)
        log_sum_pdf = torch.log(sum_pdf)
        loss = -torch.mean(log_sum_pdf)

        return loss
