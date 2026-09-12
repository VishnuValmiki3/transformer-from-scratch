import math

import torch

from tfs.optim import clip_grad_norm_, cosine_lr

SCHEDULE = dict(max_lr=1e-3, min_lr=1e-4, warmup_steps=100, cosine_steps=1000)


def test_warmup_is_linear_from_zero():
    assert cosine_lr(0, **SCHEDULE) == 0.0
    assert math.isclose(cosine_lr(50, **SCHEDULE), 5e-4)


def test_peaks_at_end_of_warmup():
    assert math.isclose(cosine_lr(100, **SCHEDULE), SCHEDULE["max_lr"], rel_tol=1e-9)


def test_decays_to_min_lr_at_end_of_cycle():
    assert math.isclose(cosine_lr(1000, **SCHEDULE), SCHEDULE["min_lr"], rel_tol=1e-9)


def test_stays_at_min_lr_after_cycle():
    assert math.isclose(cosine_lr(5000, **SCHEDULE), SCHEDULE["min_lr"], rel_tol=1e-9)


def test_cosine_phase_decreases_monotonically():
    lrs = [cosine_lr(step, **SCHEDULE) for step in range(100, 1001, 25)]
    assert all(later < earlier for earlier, later in zip(lrs, lrs[1:]))


def test_halfway_through_cosine_is_midpoint_lr():
    midpoint = (SCHEDULE["max_lr"] + SCHEDULE["min_lr"]) / 2
    assert math.isclose(cosine_lr(550, **SCHEDULE), midpoint, rel_tol=1e-9)


def _params_with_grads(values):
    params = []
    for value in values:
        p = torch.nn.Parameter(torch.zeros(len(value)))
        p.grad = torch.tensor(value)
        params.append(p)
    return params


def test_leaves_small_gradients_untouched():
    params = _params_with_grads([[0.3, 0.4]])  # norm 0.5
    clip_grad_norm_(params, max_norm=1.0)
    assert torch.allclose(params[0].grad, torch.tensor([0.3, 0.4]))


def test_scales_large_gradients_to_max_norm():
    params = _params_with_grads([[3.0, 4.0]])  # norm 5.0
    clip_grad_norm_(params, max_norm=1.0)
    assert math.isclose(params[0].grad.norm().item(), 1.0, rel_tol=1e-4)


def test_clips_on_the_global_norm_across_all_parameters():
    params = _params_with_grads([[3.0], [4.0]])  # global norm 5.0, each below max alone
    clip_grad_norm_(params, max_norm=1.0)
    global_norm = torch.cat([p.grad for p in params]).norm().item()
    assert math.isclose(global_norm, 1.0, rel_tol=1e-4)


def test_preserves_gradient_direction():
    params = _params_with_grads([[3.0, 4.0]])
    clip_grad_norm_(params, max_norm=1.0)
    assert torch.allclose(params[0].grad, torch.tensor([0.6, 0.8]), atol=1e-4)


def test_returns_pre_clip_norm_for_logging():
    params = _params_with_grads([[3.0, 4.0]])
    assert math.isclose(clip_grad_norm_(params, max_norm=1.0), 5.0, rel_tol=1e-5)


def test_ignores_parameters_without_gradients():
    params = _params_with_grads([[3.0, 4.0]])
    params.append(torch.nn.Parameter(torch.ones(2)))  # grad is None
    assert math.isclose(clip_grad_norm_(params, max_norm=10.0), 5.0, rel_tol=1e-5)


def test_returns_zero_when_nothing_has_gradients():
    assert clip_grad_norm_([torch.nn.Parameter(torch.ones(2))], max_norm=1.0) == 0.0
