#%%
from deepLFM.features import NPFeatures
import torch
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal, kl_divergence


# %%
z = torch.linspace(0, 1, 50)
mu = 10 * torch.ones(50, requires_grad=True)
LK = 0.1 * torch.eye(50, requires_grad=True)
u_dist = MultivariateNormal(mu, scale_tril=LK)


features = NPFeatures(u_dist, z, 1.0, 1.0, 1.0, 100, 10)

ls = torch.tensor([0.2], requires_grad=True)
features.ls = ls
K = features.compute_K()
ans = torch.sum(K[0])
ans.backward()
ls.grad


x = torch.linspace(-1, 1, 200)
samps = features.sample_G(x)
plt.plot(x, samps.T.detach().numpy())
# %%


kl_divergence(u_dist, u_dist)
# %%
features.Nu
# %%
