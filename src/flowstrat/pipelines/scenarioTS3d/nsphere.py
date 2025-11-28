import numpy as np
from scipy.integrate import quad
from scipy.special import beta


def cartesian2shperical(points):
    """
    Change d-dim cartesian point cloud into the d-dim spherical coordinates.
    Inverse of _spherical2cartesian
    :param points:
    :return:
    """
    # https://en.wikipedia.org/wiki/N-sphere Section: Spherical coordinates
    coords = np.zeros(shape=points.shape)
    dim = points.shape[1]
    coords[:, 0] = np.linalg.norm(points, axis=1)  # r
    for i in range(1, dim - 1):
        coords[:, i] = np.arccos(
            points[:, i - 1] / np.linalg.norm(points[:, (i - 1) :], axis=1)
        )
    id_negative = points[:, dim - 1] < 0
    coords[:, dim - 1] = np.arccos(
        points[:, dim - 2] / np.sqrt(points[:, dim - 2] ** 2 + points[:, dim - 1] ** 2)
    )
    # change cooridnate if negative
    coords[id_negative, dim - 1] = 2 * np.pi - coords[id_negative, dim - 1]
    return coords


def spherical2cartesian(points):
    # https://en.wikipedia.org/wiki/N-sphere Section: Spherical coordinates
    coords = np.zeros(shape=points.shape)
    dim = points.shape[1]
    coords[:, 0] = points[:, 0] * np.cos(points[:, 1])
    temp = points[:, 0]
    for d in range(1, dim - 1):
        temp = temp * np.sin(points[:, d])
        coords[:, d] = temp * np.cos(points[:, d + 1])
    coords[:, dim - 1] = temp * np.sin(points[:, dim - 1])
    return coords


# code related to distribution which density is sin^d(x)
def sind_normalizarion_constant(d):
    """
    Returns normalization factor for sin^d distribution

    :param d: dimension
    :return:
    """
    # def sindensity(x):
    #     return np.sin(x) ** d
    #
    # norm = quad(sindensity, 0, np.pi)[0]
    # return norm
    return beta(0.5, (d + 1) / 2)


# Rejection sampling
# https://cosmiccoding.com.au/tutorials/rejection_sampling
# https://math.stackexchange.com/questions/2682971/what-is-the-distribution-of-angles-for-a-point-uniform-on-the-unit-n-sphere
def sind_sample(n, d, low=0, high=np.pi, batch=1000):
    """
    Draw sample from distribution which density is ~sin^d(x) over [0, pi]
    :param n: number of points to be drawn
    :param d: d
    :param low: left end of an interval
    :param high: right end of an interval
    :param batch:
    :return:
    """
    # maxixmum value of ~sin**d density depends on normalization constant only
    # as max of sin(x)**d function on [0, pi] is 1
    norm = sind_normalizarion_constant(d)
    ymax = 1 / norm
    samples = []
    while len(samples) < n:
        x = np.random.uniform(low=low, high=high, size=batch)
        y = np.random.uniform(low=0, high=ymax, size=batch)
        samples += x[y < (np.sin(x) ** d) / norm].tolist()
    return samples[:n]


def sind_quantile(q, d):
    """
    Distribution of phi angles
        ~sin for last phi angle (out of those defined on 0,pi)
        ~sin^2 for -1 last angle
        ~sin^3 for -2 last angle
    Compute the quantiles of that distribution

    :param q: qvalue
    :param d: dimension
    :return:
    """
    ts = np.linspace(0, np.pi, 10000)

    def sindensity(x):
        return np.sin(x) ** d

    norm = sind_normalizarion_constant(d)
    q_diff = [np.abs(quad(sindensity, 0, t)[0] / norm - q) for t in ts]
    idx = np.argmin(q_diff)
    return ts[idx]
