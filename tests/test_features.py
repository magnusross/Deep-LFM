from deepLFM import features
import torch


def test_I1():
    ans = torch.complex(torch.Tensor([2.53379]), torch.Tensor([0.514595]))
    ours = features.I1(
        torch.Tensor([0.1]),
        torch.Tensor([0.2]),
        torch.Tensor([0.3]),
        torch.Tensor([0.4]),
        torch.Tensor([0.5]),
    )
    assert torch.isclose(ans, ours)


def test_I2():
    ans = torch.complex(torch.Tensor([2.03472]), torch.Tensor([-0.101821]))
    ours = features.I2(
        torch.Tensor([0.1]),
        torch.Tensor([0.2]),
        torch.Tensor([0.3]),
        torch.Tensor([0.4]),
        torch.Tensor([0.5]),
    )
    assert torch.isclose(ans, ours)
