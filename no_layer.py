#%%
from deepLFM.features import NPFeatures
import torch
import matplotlib.pyplot as plt
from deepLFM.utils import JITTER

#%%

Nu = 5
z = torch.linspace(-5.0, 5.0, Nu)
alpha = 0.1
f_ls = 2.0
f_amp = 1.0
f_Nbasis = 30
f_Ns = 10

features = NPFeatures(z, alpha, f_ls, f_amp, f_Nbasis, f_Ns)
tgs = torch.linspace(-5, 5, 100)
samples = features.sample_G(tgs)
plt.plot(tgs, samples.T.detach().numpy())

# %%
ls = 0.5
Nbasis = 50
Ns = 10
ts = torch.linspace(-10, 10, 200)


def sample_omegas(Nbasis, ls):
    return torch.randn(Nbasis) / ls


omegas = sample_omegas(Nbasis, ls)
phi = features.sample_features(ts.reshape(1, -1), omegas, Ns)


def phi2phic(phi):
    return torch.cat([phi.real, phi.imag], axis=2)


phic = phi2phic(phi)

# %%
Kff = torch.matmul(phic, phic.transpose(1, 2))


# %%
for i in range(5):  # Kff.shape[0]):
    mv_normal = torch.distributions.MultivariateNormal(
        torch.zeros(200), Kff[i] + 1e-2 * torch.eye(Kff.shape[1])
    )
    plt.plot(ts, mv_normal.sample())
plt.show()

# %%
