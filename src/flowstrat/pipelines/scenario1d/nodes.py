from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torch.nn

import logging

from flowstrat.pipelines.node_utils import train_cnf_model

LOGGER = logging.getLogger(__name__)


def sample_data_1D(n_samples: int):
    """Sample from 1D distribution introduced in Example 1."""
    n_samples1, n_samples2, n_samples3 = (
        int(n_samples / 4),
        int(n_samples / 2),
        int(n_samples / 4),
    )
    x = np.random.beta(7, 1.1, size=(n_samples1, 1))
    y = np.random.uniform(low=0.2, high=0.4, size=(n_samples2, 1))
    z = np.random.normal(loc=0.6, scale=0.06782329983125268, size=(n_samples3, 1))
    return np.concatenate([x, y, z])


def sample_from_flow_model_using_1d_grid_points(
    model, R: int, m: int, layer_sizes=None, method: str = None
) -> Dict[int, np.ndarray]:
    """
    Generate R samples from target distribution using stratification with proportional weights.

    Args:
        model: pre-trained flow model
        R: number of samples to generate
        m: number of grid points
        layer_sizes: number of samples to generate for each grid point
    """
    from scipy.stats import norm

    grid = np.linspace(0.0, 1.0, m + 1)

    if layer_sizes is None:
        assert R % m == 0
        layer_sizes = defaultdict(lambda: R // m)
    #     else:
    #         assert R == sum(Rij.values())

    x = dict()
    with torch.no_grad():
        for i in range(m):
            z_x = norm.ppf(np.random.uniform(grid[i], grid[i + 1], layer_sizes[i]))
            x[i] = (
                model.forward(
                    torch.tensor(z_x, dtype=torch.float).reshape(-1, 1), reverse=True
                )
                .detach()
                .numpy()
            )
    return x


def get_target_functions_and_estimations():
    target_functions = {
        "rho1": (lambda x1: np.sin(np.exp(x1))),
        "rho2": (lambda x1: np.log(1.0 + np.abs(x1))),
        "rho3": (lambda x1: 1.0 / np.log(1.0 + np.abs(x1))),  # not defined at 0
        "gt_0.95": (lambda x1: x1 >= 0.95),
        "lt_0.95": (lambda x1: x1 < 0.95),
        "lt_0.99": (lambda x1: x1 < 0.99),
    }

    target_estimations = {
        "rho1": 0.8919046590,
        "rho2": 0.4031789897,
        "rho3": 2.917446964,
        "gt_0.95": 0.06489696896,
        "lt_0.95": 0.9351030310,
        "lt_0.99": 0.9874830473,
    }
    return target_functions, target_estimations
