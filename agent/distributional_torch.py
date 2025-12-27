from absl import flags
from typing import Optional, Sequence, Union
import enum
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
from scipy.stats import norm

FLAGS = flags.FLAGS


def quantile_project(q, v, q_grid)-> torch.Tensor:
    """Project quantile distribution (quantile_grid, values) onto quantile under the L2-metric over CDFs.

    This projection works for any support q.
    Let Kq be len(q_grid)

    Args:
    q:  quantile
    v: (batch_size, Kq) values to project onto
    q_grid:  (Kq,) Quantiles for P(Zp[i])

    Returns:
    Quantile projection of (q_grid, v) onto q.
    """

    # Asserts that Zq has no leading dimension of size 1.
    if q_grid.dim() > 1:
        q_grid = q_grid.squeeze(0)

    q = torch.tensor(q, dtype=v.dtype, device=v.device)

    # Extracts vmin and vmax and construct helper tensors from Zq.
    vmin, vmax = q_grid[0], q_grid[-1]
    d_pos = torch.cat([q_grid[1:], vmin.unsqueeze(0)], dim=0) - q_grid
    d_neg = q_grid - torch.cat([vmax.unsqueeze(0), q_grid[:-1]], dim=0)

    # clip q
    clipped_q = torch.clamp(q, vmin, vmax)
    eq_mask = (q_grid == q).float()
    if eq_mask.sum() == 1:
        idx = eq_mask.nonzero(as_tuple=True)[0].item()
        return v[:, idx]  # (batch,)

    # Expand dims for broadcasting
    d_pos = d_pos.unsqueeze(0)  # (1, Kq)
    d_neg = d_neg.unsqueeze(0)  # (1, Kq)
    q_grid_expand = q_grid.unsqueeze(0)  # (1, Kq)
    delta_qp = clipped_q - q_grid_expand  # (1, Kq)
    d_sign = (delta_qp >= 0).float()  # (1, Kq)

    # Linear interpolation weights
    delta_hat = d_sign * delta_qp / d_pos - (1 - d_sign) * delta_qp / d_neg
    weight = torch.clamp(1 - delta_hat, 0, 1)  # (1, Kq)

    # Weighted sum
    result = torch.sum(weight * v, dim=1)  # (batch,)
    return result


class QuantileDistribution:
    def __init__(self,values, quantiles, probs,
                 name: str = 'QuantileDistribution'):
        """Quantile Distribution
        values: (batch_size, Kq)
        quantiles: (Kq,)
        probs: (Kq,)
        """
        self.values = values
        self.quantiles = quantiles
        self.probs = probs
        self.name = name
        # For sampling convenience:
        # expand probs -> (batch_size, Kq)
        self._expanded_probs = None
        self.lambda_cvar = 0.2

    def _get_expanded_probs(self):
        """Expand probabilities to (batch_size, Kq) for multinomial sampling."""
        if self._expanded_probs is None:
            batch = self.values.shape[0]
            # repeat probs for batch dimension
            self._expanded_probs = self.probs.unsqueeze(0).expand(batch, -1)
        return self._expanded_probs

    def sample(self, sample_shape=torch.Size()):
        """
        Return quantile values sampled according to probs.
        sample_shape: torch.Size([n]) -> return (n, batch)
        """
        probs = self._get_expanded_probs()  # (B, Kq)

        if len(sample_shape) == 0:
            # Single sample per batch
            idx = torch.multinomial(probs, 1).squeeze(1)  # (B,)
            out = self.values.gather(1, idx.unsqueeze(-1)).squeeze(-1)
            return out

    def mean(self):
        mu = (self.values * self.probs).sum(dim=-1)
        return mu

    def variance(self):
        m = self.mean().unsqueeze(-1)
        return ((self.values - m)**2 * self.probs).sum(dim=-1)

    def stddev(self):
        return self.variance().sqrt()

    def meanstd(self):
        """Implements mean-volc*std"""
        volc = FLAGS.std_coef
        return self.mean() - volc * self.stddev()

    def var(self, th: float) -> torch.Tensor:
        """
        Value-at-Risk at confidence level th.
        th: float, e.g., 0.95 for left tail 5%
        """
        return quantile_project(1 - th, self.values, self.quantiles)

    def meanvar(self,
                alpha: float = 0.2,
                tail_frac: float = 0.1,
                temperature: float = 0.2 ) -> torch.Tensor:
        Q = self.values  # (B, K)
        taus = self.quantiles  # (K,)
        mean_Q = Q.mean(dim=1)  # (B,)
        K = Q.shape[1]
        k_tail = int(tail_frac * K)
        Q_tail = Q[:, :k_tail]  # (B, k)
        taus_tail = taus[:k_tail]  # (k,)
        # emphasize smaller tau but smoothly
        scores = -(taus_tail - taus_tail.mean())
        weights = torch.softmax(scores / temperature, dim=0)  # (k,)
        tail_Q = (Q_tail * weights.unsqueeze(0)).sum(dim=1)
        return (mean_Q + alpha * tail_Q).mean()

    def weighted_tail_loss(
            self,
            tail: str = 'left',
            temperature: float = 0.1
    ):
        """
        Smoothed tail-weighted actor loss using softmax weighting.
        """

        Q = self.values  # (B, K)
        taus = self.quantiles  # (K,)

        if tail == 'left':
            scores = -taus  # smaller tau -> larger weight
        elif tail == 'right':
            scores = taus
        else:
            raise ValueError

        # temperature smoothing
        weights = torch.softmax(scores / temperature, dim=0)  # (K,)
        weights = weights.unsqueeze(0)  # (1, K)

        weighted_Q = (weights * Q).sum(dim=1)  # (B,)
        return weighted_Q.mean()

    def cvar(self, th: float) -> torch.Tensor:
        quantile = 1 - th
        cdf = torch.cumsum(self.probs, dim=-1)  # (Kq,)
        mask = cdf <= quantile
        cprobs = torch.where(mask, self.probs, torch.zeros_like(self.probs))  # (Kq,)
        return torch.sum(cprobs * self.values, dim=-1)  # (batch_size,)


class QuantileDistProbType(enum.Enum):
    LEFT = 1
    MID = 2
    RIGHT = 3



class QuantileDiscreteValuedHead(nn.Module):
    """
    receive critic output and transform it into distribution.
    """
    def __init__(self,
                 quantiles: np.ndarray,
                 prob_type: QuantileDistProbType = QuantileDistProbType.MID,
                ):
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)
        assert quantiles[0] > 0
        assert quantiles[-1] < 1.0
        left_probs = quantiles - np.insert(quantiles[:-1], 0, 0.0)
        right_probs = np.insert(
            quantiles[1:], len(quantiles)-1, 1.0) - quantiles
        if prob_type == QuantileDistProbType.LEFT:
            probs = left_probs
        elif prob_type == QuantileDistProbType.MID:
            probs = (left_probs + right_probs) / 2
        elif prob_type == QuantileDistProbType.RIGHT:
            probs = right_probs
        self.probs = torch.tensor(probs, dtype=torch.float32)

    def forward(self, quantile_values: torch.Tensor):
        """
        quantile_values: (batch_size, Kq)
        returns a QuantileDistribution
        """

        quantiles = self.quantiles.to(quantile_values.device).to(quantile_values.dtype)
        probs = self.probs.to(quantile_values.device).to(quantile_values.dtype)
        return QuantileDistribution(values=quantile_values, quantiles=quantiles,
                                    probs=probs)

class QuantileLoss(nn.Module):
    def __init__(self, loss_type='huber', b_decay=0.9 ):
        super().__init__()
        self.loss_type = loss_type
        self.b_decay = b_decay
        self.register_buffer('b', torch.tensor(1.0))

    def huber(self, x, k=1.0):
        return torch.where(torch.abs(x) < k, 0.5 * x ** 2, k * (torch.abs(x) - 0.5 * k))


    def gaussian_loss(self, td_error, b):
        abs_u = torch.abs(td_error)
        phi = 0.5 * (1 + torch.erf(-abs_u / b / math.sqrt(2.0)))
        loss = abs_u * (1 - 2 * phi) + b * math.sqrt(2.0 / math.pi) * torch.exp(-abs_u**2 / (2 * b**2)) - b * math.sqrt(2.0 / math.pi)
        return loss

    def gaussian_loss_taylor(self, td_error, b):
        abs_u = torch.abs(td_error)
        loss = torch.where(abs_u <= b, abs_u**2 / (b * math.sqrt(2.0 * math.pi)), abs_u - b * math.sqrt(2.0 / math.pi))
        return loss

    def laplace_loss(self, td_error, b):
        abs_u = torch.abs(td_error)
        loss = abs_u + b * torch.exp(-abs_u / b) - b
        return loss

    def laplace_loss_taylor(self, td_error, b):
        abs_u = torch.abs(td_error)
        loss = torch.where(abs_u <= b, abs_u**2 / (2*b), abs_u - b)
        return loss

    def  forward(self, q_tm1, r_t, d_t, q_t):
        """Implements Quantile Regression Loss
        q_tm1: QuantileDistribution at t-1
        r_t:   reward tensor, shape (batch,)
        d_t:   discount tensor, shape (batch,)
        q_t:   QuantileDistribution at t (target)
        loss_type: 'huber', 'gl', 'gl-tl', 'lapl', 'lapl-tl'
        """

        z_t = r_t.view(-1,1) + d_t.view(-1,1) * q_t.values  # (batch, Kq)
        z_tm1 = q_tm1.values  # (batch, Kq)
        # diff shape: (batch, Kq, Kq_target)
        diff = z_t.unsqueeze(1) - z_tm1.unsqueeze(2)

        if self.loss_type == 'huber':
            k = 1
            loss = self.huber(diff, k) / k

        # shape broadcasting: diff_detach < 0 -> 0/1 mask, has to be on Kq_target dim
        weight = torch.abs(q_tm1.quantiles.view(1, -1, 1) - (diff < 0).float())

        loss = loss * weight
        return loss.mean()

