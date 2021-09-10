#%%
from deepLFM.features import NPFeatures
from deepLFM.deepLFM import deepLFM
import torch
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal, kl_divergence

# %%
z = torch.linspace(-2, 2, 5)
mu = 1.0 * z
LK = 0.01 * torch.eye(5, requires_grad=True)
u_dist = MultivariateNormal(mu, scale_tril=LK)

Nu = 5
Ns = 10
Nt = 12
Ntom = 13
Nmag = 14

# %%
kl_divergence(u_dist, u_dist)
# %%

# %%
# ts = torch.linspace(-1, 1, 50)
# model = deepLFM(
#     1,
#     1,
#     n_hidden_layers=1,
#     n_lfm=1,
#     n_rff=1,
#     n_lf=1,
#     mc=10,
#     q_Omega_fixed_epochs=0,
#     q_theta_fixed_epochs=0,
#     local_reparam=True,
#     feed_forward=False,
# )
# model.N = 50
# # model.mc = 10
# samples = model.predict(ts)
# #%%
# samples.shape
# # %%
# plt.plot(ts, samples[0][:, 0].T.detach().numpy())
# plt.fill_between(
#     ts,
#     samples[0][:, 0].T.detach().numpy() - samples[1][:, 0].T.detach().numpy(),
#     samples[0][:, 0].T.detach().numpy() + samples[1][:, 0].T.detach().numpy(),
#     alpha=0.1,
# )
# # %%
# tgs = torch.linspace(-2, 2, 100)
# g_samples = model.features.sample_G(tgs)
# plt.plot(tgs, g_samples.T.detach().numpy())
# %%
