import pytest
import torch

from tfs.optim import AdamW


def _run_optimizers(steps=25, lr=1e-2, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
    """Drive our AdamW and torch.optim.AdamW over identical params/grads."""
    torch.manual_seed(0)
    init = torch.randn(10, 4)
    target = torch.randn(10, 4)

    ours = torch.nn.Parameter(init.clone())
    theirs = torch.nn.Parameter(init.clone())

    opt_ours = AdamW([ours], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    opt_theirs = torch.optim.AdamW([theirs], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    for _ in range(steps):
        for param, opt in ((ours, opt_ours), (theirs, opt_theirs)):
            opt.zero_grad()
            ((param - target) ** 2).sum().backward()
            opt.step()
    return ours, theirs


def test_matches_torch_adamw_with_weight_decay():
    ours, theirs = _run_optimizers(weight_decay=0.01)
    assert torch.allclose(ours, theirs, atol=1e-6)


def test_matches_torch_adamw_without_weight_decay():
    ours, theirs = _run_optimizers(weight_decay=0.0)
    assert torch.allclose(ours, theirs, atol=1e-6)


def test_matches_torch_adamw_with_nondefault_betas():
    ours, theirs = _run_optimizers(betas=(0.8, 0.95), lr=3e-3)
    assert torch.allclose(ours, theirs, atol=1e-6)


def test_converges_on_a_convex_quadratic():
    param = torch.nn.Parameter(torch.tensor([5.0, -3.0]))
    opt = AdamW([param], lr=0.1, weight_decay=0.0)
    for _ in range(500):
        opt.zero_grad()
        (param**2).sum().backward()
        opt.step()
    assert torch.allclose(param, torch.zeros(2), atol=1e-3)


def test_first_step_magnitude_is_about_lr_regardless_of_gradient_scale():
    """Bias correction makes Adam's first step ~lr even for a tiny or huge gradient."""
    for grad_scale in (1e-4, 1e4):
        param = torch.nn.Parameter(torch.zeros(3))
        opt = AdamW([param], lr=0.1, weight_decay=0.0)
        param.grad = torch.full((3,), grad_scale)
        opt.step()
        assert torch.allclose(param.abs(), torch.full((3,), 0.1), rtol=1e-3)


def test_decoupled_decay_shrinks_weights_when_gradient_is_zero():
    param = torch.nn.Parameter(torch.ones(4))
    opt = AdamW([param], lr=0.1, weight_decay=0.5)
    param.grad = torch.zeros(4)
    opt.step()
    # Pure decay: p <- p * (1 - lr * wd); the Adam term contributes nothing at zero grad.
    assert torch.allclose(param, torch.full((4,), 0.95), atol=1e-6)


def test_state_is_created_lazily_and_tracks_step_count():
    param = torch.nn.Parameter(torch.ones(2))
    opt = AdamW([param], lr=0.1)
    assert opt.state[param] == {}

    param.grad = torch.ones(2)
    opt.step()
    opt.step()
    assert opt.state[param]["step"] == 2
    assert opt.state[param]["exp_avg"].shape == param.shape


def test_skips_parameters_without_gradients():
    used = torch.nn.Parameter(torch.ones(2))
    unused = torch.nn.Parameter(torch.ones(2))
    opt = AdamW([used, unused], lr=0.1, weight_decay=0.1)
    used.grad = torch.ones(2)
    opt.step()
    assert torch.allclose(unused, torch.ones(2))


@pytest.mark.parametrize(
    "kwargs",
    [dict(lr=-1.0), dict(betas=(1.0, 0.999)), dict(eps=-1e-8), dict(weight_decay=-0.1)],
)
def test_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        AdamW([torch.nn.Parameter(torch.ones(1))], **kwargs)
