import torch
from .utils import JITTER
from math import pi


def I1(
    t: torch.Tensor,
    alph: torch.Tensor,
    thet: torch.Tensor,
    beta: torch.Tensor,
    omeg: torch.Tensor,
) -> torch.Tensor:
    """
    Computes integal of random part of G with e^{jwt}.
    Inputs must have same shape.

    Args:
        t (torch.Tensor): input times.
        alph (torch.Tensor): G window size.
        thet (torch.Tensor): Random basis frequencies (sampled from PSD of kernel).
        beta (torch.Tensor): Random basis phases (sampled from U).
        omeg (torch.Tensor): Random features frequecies.

    Returns:
        torch.Tensor: Value of the inegral.
    """

    const = 0.5 * torch.sqrt(pi / alph)
    t1 = torch.exp(torch.complex(-((thet + omeg) ** 2) / (4 * alph), t * omeg - beta))
    t2 = 1 + torch.exp(torch.complex(thet * omeg / alph, 2 * beta))
    return const * t1 * t2


def I2(
    t: torch.Tensor,
    alph: torch.Tensor,
    z: torch.Tensor,
    p: torch.Tensor,
    omeg: torch.Tensor,
) -> torch.Tensor:
    """
    Computes integal of canonical basis part of G with e^{jwt}.
    Inputs must have same shape.


    Args:
        t (torch.Tensor): input times.
        alph (torch.Tensor): G window size.
        z (torch.Tensor): Inducing point inputs.
        p (torch.Tensor): Kernel precision.
        omeg (torch.Tensor): Random features frequecies.
    Returns:
        torch.Tensor: Value of the integral.
    """
    const = torch.sqrt(pi / (p + alph))
    t1 = omeg * torch.complex(omeg, -4 * p * (t - z))
    t2 = 4 * alph * torch.complex(p * z ** 2, -t * omeg)
    return const * torch.exp(-(t1 + t2) / (4 * (alph + p)))


class NPFeatures:
    def __init__(self, u_dist, z, alph, ls, amp, Nbasis, Ns):
        self.u_dist = u_dist
        self.Nu = u_dist.sample().shape[0]

        self.z = z
        self.alph = alph
        self.ls = ls
        self.amp = amp
        self.Nbasis = Nbasis
        self.Ns = Ns
        self.set_K()

    def compute_K(self, t1s=None, t2s=None):
        if t1s is None and t2s is None:
            t1s = self.z
            t2s = self.z
        elif t2s is None:
            t2s = t1s

        z_i = t1s[:, None]
        z_j = t2s[None, :]
        D_ij = (1 / self.ls ** 2) * (z_i - z_j) ** 2
        K = self.amp * torch.exp(-D_ij)

        if t1s.shape[0] == t2s.shape[0]:
            LK = torch.linalg.cholesky(K + JITTER * torch.eye(K.shape[0]))
        else:
            LK = None
        return K, LK

    def set_K(self):
        K, LK = self.compute_K()
        self.K = K
        self.LK = LK

    def sample_basis(self):
        thets = torch.randn(self.Ns, self.Nbasis) / self.ls
        ws = (
            self.amp
            * torch.sqrt(torch.tensor(2.0 / self.Nbasis, requires_grad=False))
            * torch.randn(self.Ns, self.Nbasis)
        )
        betas = 2 * pi * torch.rand(self.Ns, self.Nbasis)
        return thets, betas, ws

    def compute_Phi(self, thets, betas, ws, ts=None):
        if ts is None:
            ts = self.z
        N = ts.shape[0]
        rthets = thets.unsqueeze(-1).repeat(1, 1, N)
        rbetas = betas.unsqueeze(-1).repeat(1, 1, N)
        rts = ts.unsqueeze(0).unsqueeze(0).repeat(self.Ns, self.Nbasis, 1)
        return torch.cos(rthets * rts + rbetas)

    def compute_q(self, thets, betas, ws):

        phi = self.compute_Phi(thets, betas, ws)
        us = self.u_dist.rsample(sample_shape=(self.Ns,))
        rLK = self.K.unsqueeze(0).repeat(self.Ns, 1, 1)
        x = us.unsqueeze(-1) - phi.transpose(1, 2).matmul(ws.unsqueeze(-1))
        return torch.cholesky_solve(x, rLK)

    def sample_G(self, ts):
        thets, betas, ws = self.sample_basis()
        qs = self.compute_q(thets, betas, ws)
        phis = self.compute_Phi(thets, betas, ws, ts=ts)
        K, _ = self.compute_K(t1s=ts, t2s=self.z)
        print(K.shape)
        print(qs.shape)
        basis_part = qs.squeeze(-1).matmul(K.T)
        rws = ws.unsqueeze(-1).repeat(1, 1, ts.shape[0])
        random_part = torch.sum(rws * phis, 1)

        return random_part + basis_part
