#%%
from deepLFM.features import NPFeatures
import torch
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal, kl_divergence

# %%
z = torch.linspace(-2, 2, 5)
mu = 1.0 * z
LK = 0.01 * torch.eye(5, requires_grad=True)
u_dist = MultivariateNormal(mu, scale_tril=LK)

Ns = 10
Nt = 12
Ntom = 13
Nmag = 14

ls = torch.tensor([0.2], requires_grad=True)
features = NPFeatures(u_dist, z, 1.0, ls, 1.0, Nmag, Ns)

#
# #

ts = torch.randn(Ns, Nt)
omegas = torch.randn(Ntom)
features.sample_features(ts, omegas, Ns)
# %%
kl_divergence(u_dist, u_dist)
# %%
features.Nu
# %%
