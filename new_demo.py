#%%
from deepLFM.features import NPFeatures
import torch
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal, kl_divergence

# %%
z = torch.linspace(-2, 2, 5)
mu = 1.0 * z
LK = 0.0000001 * torch.eye(5, requires_grad=True)
u_dist = MultivariateNormal(mu, scale_tril=LK)


ls = torch.tensor([0.2], requires_grad=True)
features = NPFeatures(u_dist, z, 1.0, ls, 1.0, 98, 99)

#
# #


x = torch.linspace(-3, 3, 100)
samps = features.sample_G(x)
ans = torch.sum(samps)
ans.backward()
print(ls.grad)

plt.plot(x, samps.T.detach().numpy())


# %%
kl_divergence(u_dist, u_dist)
# %%
features.Nu
# %%
