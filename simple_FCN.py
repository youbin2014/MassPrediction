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
    def __init__(self):
        super(FCN_Det, self).__init__()

        self.fc1 = nn.Linear(
            in_features=3,
            out_features=32
        )
        self.fc2 = nn.Linear(
            in_features=32,
            out_features=1,
        )

    def forward(self, x):
        x= self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output=x
        return output


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()

        self.fc1 = LinearReparameterization(
            in_features=3,
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
