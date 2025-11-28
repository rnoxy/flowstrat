import copy
import itertools
import pickle

import scipy.stats as ss
import numpy as np
import torch
from icecream import ic
from scipy.stats import multivariate_t
import pandas as pd
import logging

from flowstrat.cnf_utils import (
    build_model_tabular,
    standard_t_logprob,
    weigthed_normal_logprob,
    standard_normal_logprob,
)
from flowstrat.pipelines.scenarioTS3d.nsphere import (
    sind_quantile,
    sind_sample,
    spherical2cartesian,
)
from flowstrat.utils import parse_arguments, set_seed_everywhere
from flowstrat.utils_quantiles import transform_by_flow


LOGGER = logging.getLogger(__name__)


def sample_distribution_observations(
    n: int, params: dict[str, any], dim: int
) -> np.ndarray:
    name = params["name"]

    if name == "uniform":
        return np.random.uniform(
            low=params["params"]["low"], high=params["params"]["high"], size=n
        )
    elif name == "beta":
        return np.random.beta(a=params["params"]["a"], b=params["params"]["b"], size=n)
    elif name == "normal":
        return np.random.normal(
            loc=params["params"]["loc"], scale=params["params"]["scale"], size=n
        )
    elif name == "tsnd":
        params_df = params["df"]
        params_correlation = params["correlation"]
        t_matrix = np.ones((dim, dim)) * params_correlation + np.eye(N=dim) * (
            1 - params_correlation
        )
        t_loc = [0] * dim
        mvt = multivariate_t(t_loc, t_matrix, df=params_df)
        sample = mvt.rvs(size=n)
        return sample
    else:
        raise ValueError(f"Unknown distribution name {name}")


def simulate_observations_multivariate(
    n: int,
    mixtures: list[float],
    distributions: list[dict],
    dim: int,
):
    n_dist = len(mixtures)
    if dim > 1:
        results = np.zeros((n, dim))
    else:
        results = np.zeros(n)

    current_pos = 0
    for i in np.arange(n_dist):
        w = mixtures[i]
        ni = int(np.floor(n * w))
        if i == (n_dist - 1):
            ni = n - current_pos
        results[current_pos : (current_pos + ni)] = sample_distribution_observations(
            ni, distributions[i], dim
        )
        current_pos += ni

    np.random.shuffle(results)
    return results


def generate_dataset(params):
    n_train = params["n_train"]
    n_test = params["n_test"]

    dim = params.get("dim", 3)

    distributions = params["distributions"]

    n_dist = len(distributions)

    mixtures = params.get("mixtures", [1 / n_dist] * n_dist)
    weights = params.get("hidden_state_weights")
    if weights is None:
        LOGGER.warning("Setting default weights to be [1, 1, ..., 1].")

    obs_train = simulate_observations_multivariate(
        n=n_train,
        mixtures=mixtures,
        distributions=distributions,
        dim=dim,
    )
    obs_test = simulate_observations_multivariate(
        n=n_test,
        mixtures=mixtures,
        distributions=distributions,
        dim=dim,
    )

    return pd.DataFrame(obs_train), pd.DataFrame(obs_test)


def train_model_and_split(dataset: pd.DataFrame, params):
    data = train_model(dataset, params)
    history = pd.DataFrame(
        data["history"], columns=["epoch", "train_loss", "validation_loss"]
    )
    return data, history


def train_model(dataset: pd.DataFrame, params):
    device = params.get("device", "cpu")

    dataset_train = dataset.sample(frac=0.8)
    dataset_valid = dataset.drop(dataset_train.index)

    observations_train = torch.tensor(
        dataset_train.to_numpy(), dtype=torch.float, device=device
    )
    observations_valid = torch.tensor(
        dataset_valid.to_numpy(), dtype=torch.float, device=device
    )

    observations_mean = torch.mean(observations_train, axis=0)
    observations_train = observations_train - observations_mean
    observations_valid = observations_valid - observations_mean

    # TODO: move args to "params" in yaml
    args = parse_arguments(command_args=[])

    dim = observations_train.shape[1]

    n_epochs = params["n_epochs"]
    lrate = params["lr"]
    base = params["base"]
    noise = float(params["noise"])
    if noise < 0:
        raise ValueError("Noise must be >=0")
    weight_decay = params.get("weight_decay", 0.00005)

    model = build_model_tabular(args, dims=dim)

    # Train model
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=lrate, weight_decay=weight_decay)

    def _get_loss(obs):
        y, delta_log_py = model(obs, torch.zeros(obs.size(0), 1).to(obs))
        if base == "t":
            log_py = standard_t_logprob(y, df=5).sum(1)
        elif base == "weighted":
            log_py = weigthed_normal_logprob(y).sum(1)
        else:
            log_py = standard_normal_logprob(y).sum(1)
        delta_log_py = delta_log_py.sum(1)
        log_px = log_py - delta_log_py
        loss = -1.0 * torch.mean(log_px)
        return loss

    loss_train = _get_loss(observations_train)
    ic(loss_train)

    history = []
    best_model = None
    best_loss = np.Inf
    best_epoch = 0
    early_stopping_patient = 50

    for it in range(n_epochs):
        optimizer.zero_grad()
        # add small noise to the training sample
        observations_train_noise = observations_train
        if n_epochs % 10 == 0:
            if noise < 1e-8:
                observations_train_noise = observations_train
            else:
                observations_train_noise = observations_train + torch.normal(
                    mean=0, std=noise, size=observations_train.size()
                )

        loss_train = _get_loss(observations_train_noise)
        loss_valid = _get_loss(observations_valid)

        loss_train.backward()  # retain_graph=True) #
        optimizer.step()

        loss_train = loss_train.item()
        loss_valid = loss_valid.item()

        if loss_valid < best_loss:
            best_loss = loss_valid
            best_epoch = it
            best_model = copy.copy(model)

        ic(it, loss_train, loss_valid, best_epoch, best_loss)
        history.append([it, loss_train, loss_valid])

        if it - best_epoch > early_stopping_patient:
            print("Early stopping break")
            break

    n = observations_train.shape[0]
    extra_data = {
        "obs_train_mean": observations_mean,
        "dim": dim,
        "n_epochs": n_epochs,
        "lrate": lrate,
        "n": n,
        "params": params,
        "dataset": dataset,
        "best_model": {
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "best_model": best_model,
        },
    }

    return dict(model=model, extra_data=extra_data, history=history)


def parse_args(command_args: list[str]):
    import argparse

    parser = argparse.ArgumentParser(description="Scenario TS3d ")
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--R", type=int, required=True, help="R")
    parser.add_argument("--Rpilot", type=int, required=True, help="R_pilot")
    parser.add_argument(
        "--function", choices=["h1", "h2", "h3"], required=True
    )  # which function to estimate
    parser.add_argument(
        "--strata_mode",
        choices=["spherical", "coordinates"],
        default="spherical",
        required=True,
    )
    parser.add_argument("--n_strata_r", type=int, default=10)
    parser.add_argument("--n_strata_phi", type=int, default=10)
    parser.add_argument("--n_strata_phi_last", type=int, default=10)
    parser.add_argument(
        "--flowbase", choices=["norm", "t3", "t5"], default="norm"
    )  # not used in simualtions
    parser.add_argument(
        "--stratatype", choices=["exact", "mc"], default="exact"
    )  # always use exact here
    args = parser.parse_args(command_args)
    return args


def evaluate_model(dataset_train, model, parameters):
    args = parse_args(command_args=parameters["args"])
    params = {
        "function": args.function,
        "flowbase": args.flowbase,
        "R_pilot": args.Rpilot,  # nr of pilot simulations (to estimate stds)
        "R": args.R,  # nr of points drawn from flow to validate the model
        "n_strata_r": args.n_strata_r,  # number of startification bins in r direction
        "n_strata_phi": args.n_strata_phi,  # number of startification bins for angles that change from [0, pi]
        "n_strata_phi_last": args.n_strata_phi_last,
        # number of startification bins for last angle that change from [0, 2pi]
        "strata_type": args.stratatype,  # exact or mc
        "strata_sampling": "theoretical",  # theoretical or empirical, empirical is not currently working!
        "strata_mode": args.strata_mode,  # coordinates or spherical
    }

    output_dict = estimate_function(
        model_dict=model, obs_train=dataset_train, params=params, dim=args.dim
    )
    print(output_dict)
    return output_dict


def estimate_function(model_dict, obs_train, params, dim: int):
    # set seed
    set_seed_everywhere(42)

    def _h1(dat):
        datmax0 = np.maximum(dat - 1, 0)
        ids = np.logical_not(np.all(datmax0 > 0, axis=1))
        datmax0 = np.sin(np.prod(dat, axis=1))
        datmax0[ids] = 0
        return datmax0

    def _h2(dat):
        datmax0 = np.maximum(dat - 1, 0)
        ids = np.logical_not(np.all(datmax0 > 0, axis=1))
        datmax0 = 1.0 / np.prod(dat, axis=1)
        datmax0[ids] = 0
        return datmax0

    def _h3(dat):
        datmax0 = np.maximum(dat - 1, 0)
        ids = np.logical_not(np.all(datmax0 > 0, axis=1))
        datmax0 = 1.0 / np.log(np.abs(np.prod(dat, axis=1)))
        datmax0[ids] = 0
        return datmax0

    def function_estimate(pts, name):
        if name == "h1":
            return _h1(pts)
        if name == "h2":
            return _h2(pts)
        if name == "h3":
            return _h3(pts)
        raise ValueError("Wrong function name")

    model = model_dict["model"]
    model.eval()

    R = params["R"]
    R_pilot = params["R_pilot"]
    function_name = params["function"]
    flowbase = params["flowbase"]
    strata_mode = params["strata_mode"]

    # Not everything is implemented
    # Check if combination of input parameters is valid
    if flowbase not in ["norm", "t3", "t5"]:
        raise ValueError(
            f"Parameter flowbase must be norm, t3 or t5. Found: {flowbase}"
        )
    strata_sampling = params["strata_sampling"]
    if strata_sampling not in ["theoretical", "empirical"]:
        raise ValueError(
            f"Parameter strata_sampling must be theoretical or empirical. Found: {strata_sampling}"
        )
    if strata_mode not in ["spherical", "coordinates"]:
        raise ValueError(
            f"Parameter strata_mode must be spherical or coordinates. Found: {strata_sampling}"
        )
    if strata_mode == "coordinates" and flowbase != "norm":
        raise ValueError(
            f"strata_mode = {strata_mode} is implemented only for flowbase = norm"
        )

    def _base_sample(flowbase, size):
        if flowbase == "norm":
            return torch.normal(0, 1, size=size)
        if flowbase == "t3":
            return torch.from_numpy(ss.t(df=3).rvs(size=size))
        if flowbase == "t5":
            return torch.from_numpy(ss.t(df=5).rvs(size=size))
        return None

    # function estimated from the training data
    # here we take 80% of data points as only 80% of provided data was used to train the model
    n_pts_train = int(obs_train.shape[0] * 0.8)
    function_data = function_estimate(
        obs_train.values[:n_pts_train, :dim], name=function_name
    )
    estimated_data = np.mean(function_data)
    variance_data = np.var(function_data, ddof=1) / n_pts_train

    # function estimation form
    base_sample = _base_sample(flowbase, size=(R, dim))
    flow_sample = transform_by_flow(model, base_sample)
    flow_sample += model_dict["extra_data"]["obs_train_mean"].numpy()
    function_sample = function_estimate(flow_sample, name=function_name)
    estimated_flow = np.mean(function_sample)
    variance_flow = np.var(function_sample, ddof=1) / R

    base_sample = _base_sample(flowbase, size=(R_pilot, dim))
    flow_sample = transform_by_flow(model, base_sample)
    flow_sample += model_dict["extra_data"]["obs_train_mean"].numpy()
    function_sample = function_estimate(flow_sample, name=function_name)

    # stratification
    # number of stratification boxes in r and phi axis
    n_strata_r = params["n_strata_r"]
    n_strata_phi = params["n_strata_phi"]
    n_strata_phi_last = params["n_strata_phi_last"]

    if strata_mode == "coordinates":
        print(
            f"!! NOTE: strata mode is {strata_mode}. Only n_strata_r={n_strata_r} will be used"
        )
        n_strata_phi = n_strata_r
        n_strata_phi_last = n_strata_r
        n_strata_total = n_strata_r**dim
    else:  # spherical
        n_strata_total = n_strata_r * n_strata_phi_last * n_strata_phi ** (dim - 2)

    strata_type = params["strata_type"]

    if strata_type not in ["exact", "mc"]:
        raise ValueError("strata_type must be exact or mc")
    if flowbase != "norm" and strata_type == "exact":
        raise ValueError("strata_type=exact possible only when flowbase is norm")

    # r_stratas depends on base distribution
    # if X \sim N(0,1) than r^2 = X_1^2+X_2^2+...+X_dim^2 \sim chi2(df=dim)
    # if X \sim t(df) than X^2 \sim F(1, df) but there is no direct formula for sum of independent F distribiuted RV
    # in general  one needs to simulate the quantiles
    def _simulate_r_strata(rv, MC=100000):
        X = rv.rvs(size=(MC, dim))
        dists = np.linalg.norm(X, axis=1)
        qs = np.quantile(dists, np.linspace(0, 1, n_strata_r + 1))
        # fix first and last value
        qs[0] = 0
        qs[-1] = np.inf
        return qs

    # stratification used if strata_mode == spherical
    # is computed also for strata_mode == coordinates
    if strata_type == "mc":
        print("rstrata MC")
        if flowbase == "normal":
            r_stratas = _simulate_r_strata(ss.norm())
        if flowbase == "t3":
            r_stratas = _simulate_r_strata(ss.t(df=3))
        if flowbase == "t5":
            r_stratas = _simulate_r_strata(ss.t(df=5))
    else:
        r_stratas = [
            np.sqrt(ss.chi2(df=dim).ppf(q)) for q in np.linspace(0, 1, n_strata_r + 1)
        ]
    # find stratas for angles
    phi_stratas = {}
    for d in range(1, dim - 1):
        phi_stratas[d] = [
            sind_quantile(q, dim - d - 1) for q in np.linspace(0, 1, n_strata_phi + 1)
        ]
    phi_stratas[dim - 1] = np.linspace(0, 2 * np.pi + 1e-5, n_strata_phi_last + 1)

    # stratification used if strata_mode == coordinates
    # is computed also for strata_mode == spherical although it is not required
    x_stratas = [ss.norm().ppf(q) for q in np.linspace(0, 1, n_strata_r + 1)]
    # last angle, that changes from 0 do 2pi is distributed uniformly
    if strata_mode == "spherical":
        print(f"stratas r      : {r_stratas}")
        print(f"stratas phi    : {phi_stratas}")
    else:  # coordinates
        print(f"stratas x      : {x_stratas}")

    # get ids of each stratification interval
    def _get_spherical_strata_ids(sample):
        """
        Return a label for each stratification box for spherical strata

        :param sample: data points in spherical coordinates
        :return:
        """
        ids_r = np.digitize(sample[:, 0], bins=r_stratas).reshape(-1, 1)
        ids = [ids_r]
        for d_ in range(1, dim):
            ids_phi = np.digitize(sample[:, d_], bins=phi_stratas[d_])
            ids.append(ids_phi.reshape(-1, 1))
        ids = np.concatenate(ids, axis=1)
        return ids

    def _extend_dict(dic, value=0):
        # generate all possible strata-ids
        list_of_ids = []
        list_of_ids.append(list(range(1, n_strata_r + 1)))
        for d_ in range(dim - 2):
            list_of_ids.append(list(range(1, n_strata_phi + 1)))
        list_of_ids.append(list(range(1, n_strata_phi_last + 1)))
        all_ids = list(itertools.product(*list_of_ids))
        # if some strata is not populated add it to dict with value=0
        for id_ in all_ids:
            if id_ not in dic:
                dic[id_] = value
        return dic

    def _draw_from_strata_spherical(npts, strata_id):
        # draw r - there must be a better way
        X = ss.chi2(df=dim)
        q_low = r_stratas[strata_id[0] - 1]
        q_hi = r_stratas[strata_id[0]]
        r_low = X.cdf(q_low**2)
        r_hi = X.cdf(q_hi**2)
        s_r = X.ppf(ss.uniform(loc=r_low, scale=(r_hi - r_low)).rvs(npts))
        s_r = np.sqrt(s_r)
        s_phi = []
        for d_ in range(1, dim - 1):  # starts from 1 to match keys of phi_stratas dict
            phi_low = phi_stratas[d_][strata_id[d_] - 1]
            phi_hi = phi_stratas[d_][strata_id[d_]]
            s_phi.append(sind_sample(npts, d=d_, low=phi_low, high=phi_hi))
        # draw last phi
        phi_low = phi_stratas[dim - 1][strata_id[dim - 1] - 1]
        phi_hi = phi_stratas[dim - 1][strata_id[dim - 1]]
        s_phi_last = ss.uniform(loc=phi_low, scale=(phi_hi - phi_low)).rvs(npts)
        strata_sample_sph = np.concatenate(
            [[s_r], s_phi, [s_phi_last]], axis=0
        ).transpose()
        strata_sample_cart = spherical2cartesian(strata_sample_sph)
        return strata_sample_cart

    def _draw_from_strata_coordinates(npts, strata_id):
        sx = []
        for d_ in range(dim):
            x_low = ss.norm().cdf(x_stratas[strata_id[d_] - 1])
            x_hi = ss.norm().cdf(x_stratas[strata_id[d_]])
            sx.append(
                ss.norm().ppf(ss.uniform(loc=x_low, scale=x_hi - x_low).rvs(npts))
            )
        sx = np.array(sx).transpose()
        return sx

    def _draw_from_strata(npts, strata_id):
        # return sample in cartesian coordinates!
        if strata_mode == "spherical":
            return _draw_from_strata_spherical(npts, strata_id)
        return _draw_from_strata_coordinates(npts, strata_id)

    def _function_on_strata(npts, strata_id):
        strata_sample_cart = _draw_from_strata(npts, strata_id)
        vals = transform_by_flow(model, strata_sample_cart)
        # !!! move by mean
        vals += model_dict["extra_data"]["obs_train_mean"].numpy()
        vals = function_estimate(vals, name=function_name)
        return vals

    def _compute_function_using_strata(n_pts_dict):
        function_val = 0
        function_var = 0
        # here it is assumed that probablity of each strata is the same
        p_strata = 1.0 / n_strata_total
        for key in n_pts_dict:
            n = n_pts_dict[key]
            print(f"Estimating in strata {key}, #pts {n}")
            function_values = _function_on_strata(npts=n, strata_id=key)
            function_val += np.mean(function_values) * p_strata
            if n > 1:
                function_var += (
                    np.var(function_values, ddof=1) * p_strata * p_strata / n
                )
        return function_val, function_var

    n_pts_proportional = {}
    n_pts_proportional = _extend_dict(
        n_pts_proportional, max(int(R / n_strata_total), 1)
    )
    total_std = 0
    stds = {}
    # how many point are used to estimate the STD in each strata
    n_points_in_strata_for_std = max(int(R_pilot / n_strata_total), 2)
    for id_, strata_id in enumerate(n_pts_proportional):
        if id_ % 10 == 0:
            print(f"STD est. in strata {strata_id}, #pts {n_points_in_strata_for_std}")
        values = _function_on_strata(
            npts=n_points_in_strata_for_std, strata_id=strata_id
        )
        values_std = np.std(values)
        stds[strata_id] = values_std
        total_std += values_std

    n_pts_optimal = {}
    n_pts_optimal_total = 0
    n_pts_proportional_total = 0
    for key in stds:
        n_pts_optimal[key] = max(int(stds[key] / total_std * R), 1)
        n_pts_optimal_total += n_pts_optimal[key]
        n_pts_proportional_total += n_pts_proportional[key]

    # compute normalization factors for sin^d distirbution - used only for strata_mode = spherical
    estimated_proportional, variance_proportional = _compute_function_using_strata(
        n_pts_proportional
    )
    estimated_optimal, variance_optimal = _compute_function_using_strata(n_pts_optimal)

    def _print_output(txt, val, var, alpha=0.05):
        q = ss.norm().ppf(1 - alpha / 2)
        ci_low = val - q * var
        ci_hig = val + q * var
        print(f"{txt}\t estimated={val}\t var={var}\t CI=({ci_low}, {ci_hig}")

    _print_output("estimated train data     ", val=estimated_data, var=variance_data)
    _print_output("estimated flow data      ", val=estimated_flow, var=variance_flow)
    _print_output(
        "estimated propor. strata ",
        val=estimated_proportional,
        var=variance_proportional,
    )
    _print_output(
        "estimated optimal. strata", val=estimated_optimal, var=variance_optimal
    )
    print(
        f"Total number of points drawn in prop. startification: {n_pts_proportional_total}, R={R}"
    )
    print(
        f"Total number of points drawn in opti. startification: {n_pts_optimal_total}, R={R}"
    )

    output = {
        "Yobs": {"est": estimated_data, "std:": np.sqrt(variance_data)},
        "YCMC": {"est": estimated_flow, "std:": np.sqrt(variance_flow)},
        "YPROP": {
            "est": estimated_proportional,
            "std:": np.sqrt(variance_proportional),
        },
        "YOPT ": {"est": estimated_optimal, "std:": np.sqrt(variance_optimal)},
    }

    return output
