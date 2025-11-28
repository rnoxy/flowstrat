import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn

LOGGER = logging.getLogger(__name__)


def sample_data_2D(n_samples: int):
    """Sample from 2D distribution introduced in Example 2."""
    u1 = np.random.uniform(0.0, 1.0, n_samples)
    u2 = np.random.uniform(0.0, 1.0, n_samples)
    x = -np.log(u1)
    y = -np.log(u2) / x
    result = np.array([x, y]).T
    return result


def sample_from_flow_model_using_2d_grid_points(
    model, R: int, m: int, layer_sizes=None, method="regular"
):
    """
    Calculate n_samples from target distribution using flow CNF model
    but take base samples from a grid of points in N(0, I).

    Args:
        model: flow CNF model
        R: total number of points to generate
        m: number of layers (should be a square number)
        layer_sizes: optional dictionary with number of points in each layer [i, j] (i,j=0..sqrt(m)-1).
        method: "regular" or "spherical"

    Returns:
        x: dictionary with generated points
            dictionary keys are tuples (i, j) where i,j=0..sqrt(m)-1
    """
    assert method in ["regular", "spherical"]

    def _generate_grid_points(R: int, m: int, Rij=None, method="regular"):
        """
        Generate sqrt{m} x sqrt{m} grid of points in R^2.

        Args:
            R: total number of points to generate
            m: number of layers (should be a square number)
            Rij: optional dictionary with number of points in each layer [i, j] (i,j=0..sqrt(m)-1).
            method: "regular" or "spherical"

        Returns:
            x: dictionary with generated points
                dictionary keys are tuples (i, j) where i,j=0..sqrt(m)-1
        """
        from scipy.stats import norm

        sqrt_m = int(np.floor(np.sqrt(m)))
        assert sqrt_m**2 == m
        assert R % m == 0

        grid = np.linspace(0.0, 1.0, sqrt_m + 1)

        if Rij is None:
            Rij = defaultdict(lambda: R // m)

        x = dict()
        if method == "regular":
            for i in range(sqrt_m):
                for j in range(sqrt_m):
                    x[i, j] = np.vstack(
                        [
                            norm.ppf(
                                np.random.uniform(grid[i], grid[i + 1], Rij[i, j])
                            ),
                            norm.ppf(
                                np.random.uniform(grid[j], grid[j + 1], Rij[i, j])
                            ),
                        ]
                    ).T
        elif method == "spherical":
            from scipy.stats.distributions import chi2

            for i in range(sqrt_m):
                for j in range(sqrt_m):
                    U = np.random.uniform(grid[i], grid[i + 1], Rij[i, j])
                    V = np.random.uniform(grid[j], grid[j + 1], Rij[i, j])
                    radius = np.sqrt(chi2.ppf(V, df=2))
                    phi = 2 * np.pi * U
                    x[i, j] = np.vstack([radius * np.cos(phi), radius * np.sin(phi)]).T
        else:
            raise ValueError(f"Unknown method: {method}")

        return x

    sqrt_m = int(np.floor(np.sqrt(m)))

    while True:
        z = _generate_grid_points(R, m, layer_sizes, method=method)
        x = dict()
        with torch.no_grad():
            for i in range(sqrt_m):
                for j in range(sqrt_m):
                    x[i, j] = model.forward(
                        torch.tensor(z[i, j], dtype=torch.float), reverse=True
                    ).numpy()
        all_finite = {k: np.all(np.isfinite(xs)) for k, xs in x.items()}
        print(f"all_finite: {all_finite}")
        if all(all_finite.values()):
            break
        else:
            print(
                f"Warning: some values are not finite. Regenerating... (R={R}, m={m}, method={method})"
            )
    return x


def get_target_functions_and_estimations():
    target_functions = {
        "h1": (lambda x1, x2: (x1 > 1.0) * (x2 > 1.0) * np.sin(x1 * x2)),
        "h2": (
            lambda x1, x2: np.nan_to_num(
                (x1 > 1.0) * (x2 > 1.0) * 1.0 / (x1 * x2), nan=0.0
            )
        ),  # use nan_to_num to avoid division by zero
        "h3": (
            lambda x1, x2: np.nan_to_num(
                (x1 > 1.0) * (x2 > 1.0) * 1.0 / np.log(np.abs(x1 * x2)), nan=0.0
            )
        ),  # use nan_to_num to avoid division by zero
        "gt_0.5": (lambda x1, x2: (x1 >= 0.5) * (x2 >= 0.5)),
        "gt_1.2": (lambda x1, x2: (x1 >= 1.2) * (x2 >= 1.2)),
        "gt_2.0": (lambda x1, x2: (x1 >= 2.0) * (x2 >= 2.0)),
    }
    target_estimations = {
        "h1": 0.03332466108,
        "h2": 0.3180632849e-1,
        "h3": 0.1230435329,
        "gt_0.5": 0.3149110351,
        "gt_1.2": 0.03243694071,
        "gt_2.0": 0.0008262507256,
    }
    return target_functions, target_estimations


def get_target_functions_and_estimations_for_wind2d():
    target_functions = {
        "g1": (lambda x1, x2: (x1 > 0.01) * (x2 > 0.01)),
        "g2": (lambda x1, x2: x2 / (1 + x1**2)),
        "g3": (lambda x1, x2: np.abs(x2 * x1)),
    }
    target_estimations = {key: None for key in target_functions.keys()}
    return target_functions, target_estimations
