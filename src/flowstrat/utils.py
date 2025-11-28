import logging
import pickle
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def set_seed_everywhere(seed: int):
    """Set seed everywhere ... several times ;-)"""

    LOGGER.info(f"Setting seed value to {seed}")
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random

    random.seed(seed)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np

    np.random.seed(seed)

    # 4. Set `torch.cpu` and `torch.cuda` pseudo-random generator at a fixed value
    # see https://pytorch.org/docs/stable/notes/randomness.html#
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def average_dict_values(results: List[dict]):
    """
    Take list of dictionaries with key-value and return a dictionary with the same keys and values being the
    average of the values in the list.
    Notes. We assume that the values are either floats or dictionaries, so the function is recursive.
    """
    base_dict = results[0]
    result = dict()
    for key in base_dict.keys():
        if isinstance(base_dict[key], dict):
            result[key] = average_dict_values([r[key] for r in results])
        else:
            result[key] = np.mean([r[key] for r in results])
    return result


def accuracy(x: float, x_true: Optional[float]) -> Optional[float]:
    """
    Calculate accuracy of value with respect to target.
    We measure the accuracy as the negative log of the relative error,
    i.e. the number of exact significant decimal digits.
    Args:
        x: approximation
        x_true: true value

    Returns:
        accuracy: negative log of the relative error
    """
    if x_true is None:
        return 0.0
    return -np.log10(np.abs(x - x_true) / x_true)


def sample_from_flow_model(n_samples: int, model, dim: int):
    """
    Generate n_samples from  target distribution
    using pre-trained flow CNF model.

    Notes: We use the base distribution as the prior which is a standard normal distribution.
    """
    with torch.no_grad():
        z = torch.randn(n_samples, dim)
        x = model.forward(
            z, reverse=True
        )  # we reverse in order to get samples from the target distribution
    return x.numpy()


def calculate_metrics(data: np.ndarray, g: callable, estimation_true: float):
    """
    Calculate metrics for target function g.

    We apply the function g to the `data` which is a numpy array of shape (n_samples, DIM)

    Args:
        data: samples from target distribution
    Returns: dictionary with metrics for each target function
    """
    values = g(*data.T)  # equivalent to g(data.T[:, 0], ..., data[:, dim-1])
    mean = np.mean(values)
    return dict(
        est=mean,
        std=np.std(values) / np.sqrt(len(values)),
        acc=accuracy(x=mean, x_true=estimation_true),
    )


def calculate_mean_estimations_and_variances(
    x: Dict[Any, np.ndarray],
    g_names: List[str],
    target_functions,
    target_estimations,
    normalize_variance=True,
):
    result = dict()
    for g_name in g_names:
        g = target_functions[g_name]
        # Calculate the mean of estimations
        estimator = np.mean([np.mean(g(*xs.T)) for _, xs in x.items()])
        variance = np.sum(
            [
                np.var(g(*xs.T)) / len(xs)  # s_j^2 / R_j   ; len(xs) == R_j
                for _, xs in x.items()
            ]
        )
        if normalize_variance:
            variance /= len(x) * len(x)  # m == len(x)

        result[g_name] = dict(
            est=estimator,
            std=np.sqrt(variance),
            acc=accuracy(estimator, target_estimations[g_name]),
        )
    return result


def calculate_results(
    data_train,
    model,
    n_samples,
    R,
    m,
    target_functions: Dict[str, callable],
    target_estimations: Dict[str, float],
    _methods: List[str],  # options: regular, spherical
    sample_from_model_using_grid_points: callable,
    dim: int,
    R_pilot: Optional[int] = None,
):
    g_names = target_estimations.keys()

    # 1. Calculate metrics for training data
    data = data_train
    estimations_training = {
        g_name: calculate_metrics(
            data=data,
            g=target_functions[g_name],
            estimation_true=target_estimations[g_name],
        )
        for g_name in g_names
    }

    # 2. Sample data from Flow model
    data = sample_from_flow_model(n_samples, model, dim=dim)
    estimations_sampling = {
        g_name: calculate_metrics(
            data=data,
            g=target_functions[g_name],
            estimation_true=target_estimations[g_name],
        )
        for g_name in g_names
    }

    # 3. Sampling data from Flow model using stratification with proportional weights
    estimations_proportional = dict()
    for method in _methods:
        x_init = sample_from_model_using_grid_points(
            model, R=R, m=m, layer_sizes=None, method=method
        )
        estimations_proportional[method] = calculate_mean_estimations_and_variances(
            x_init, g_names, target_functions, target_estimations
        )

    # 4. Sampling data from Flow model using stratification with weights proportional to the inverse of the variance
    # Pilot sampling
    def _calculate_estimations_optimal(method: str):
        R_prime = R_pilot if R_pilot is not None else R
        x_prime = sample_from_model_using_grid_points(
            model, R=R_prime, m=m, layer_sizes=None, method=method
        )
        estimations = dict()
        for t, g in tqdm(target_functions.items()):
            std_dev_prime = {key: np.std(g(*xs.T)) for key, xs in x_prime.items()}
            print(f"std_dev_prime: {std_dev_prime}")
            std_dev_prime_total = sum(std_dev_prime.values())
            if std_dev_prime_total > 0.0:
                print(f"std_dev_prime_total: {std_dev_prime_total}")
                layer_sizes = {
                    key: max(1, int(np.floor(s_prime / std_dev_prime_total * R)))
                    for key, s_prime in std_dev_prime.items()
                }
                x_optim = sample_from_model_using_grid_points(
                    model, R=R, m=m, layer_sizes=layer_sizes, method=method
                )
                result = calculate_mean_estimations_and_variances(
                    x_optim,
                    g_names=[t],
                    target_functions=target_functions,
                    target_estimations=target_estimations,
                )
                estimations[t] = result[t]
            else:
                print(
                    f"Warning: std_dev_prime_total = {std_dev_prime_total}. Skipping..."
                )
                estimations[t] = dict(est=0.0, std=0.0, acc=0.0)
        return estimations

    estimations_optimal = {
        method: _calculate_estimations_optimal(method) for method in _methods
    }

    return dict(
        training=estimations_training,
        sampling=estimations_sampling,
        proportional=estimations_proportional,
        optimal=estimations_optimal,
    )


def estimate(
    data_filepath: str,
    dim: int,
    model_filepath: str,
    target_functions: Dict[str, callable],
    target_estimations: Dict[str, float],
    R1: int,
    R2: int,
    m1: int,
    m2: int,
    methods: List[str],
    sample_from_model_using_grid_points: callable,
    device="cpu",
    repeats=1,
):
    # Load training data
    with open(data_filepath, "rb") as file:
        data_train = pickle.load(file)
    # Load model
    model = torch.load(model_filepath, map_location=torch.device(device))
    model.eval()

    results = []
    for _ in tqdm(range(repeats)):
        results.append(
            {
                (n, m): calculate_results(
                    data_train=data_train,
                    model=model,
                    n_samples=n,
                    R=n,
                    m=m,
                    target_functions=target_functions,
                    target_estimations=target_estimations,
                    _methods=methods,
                    sample_from_model_using_grid_points=sample_from_model_using_grid_points,
                    dim=dim,
                )
                for n in [R1, R2]
                for m in [m1, m2]
            }
        )
    return results


def print_results(
    results: dict,
    target_estimations,
    R1,
    R2,
    m1,
    m2,
    methods: List[str],
    print_latex=True,
    print_markdown=False,
    filepath: str = None,
    FACTOR=100,
):
    log2_R1 = int(np.log2(R1))
    log2_R2 = int(np.log2(R2))

    for g_name in target_estimations.keys():
        for method in methods:
            target_value = target_estimations[g_name]
            y_obs_est = results[(R1, m1)]["training"][g_name]["est"]
            y_obs_std = results[(R1, m1)]["training"][g_name]["std"]
            y_obs_acc = results[(R1, m1)]["training"][g_name]["acc"]

            def _gen_cell(metrics, best_acc, second_best_acc, type="latex"):
                """Return
                "& {est:.4f} &  {std:.4f}  & {acc:.4f}".format(est=metrics["est"], std=metrics["std"], acc=metrics["acc"])
                but underline second best and bold the best
                """
                if metrics["acc"] == best_acc:
                    text_latex = f"& {metrics['est']:.4f} &  {metrics['std'] * FACTOR:.4f}  & \\textbf{a}{metrics['acc']:.4f}{b}"
                    text_markdown = f"| {metrics['est']:.4f} |  {metrics['std'] * FACTOR:.4f}  | **{metrics['acc']:.4f}**"
                    return text_latex if type == "latex" else text_markdown
                elif metrics["acc"] == second_best_acc:
                    text_latex = f"& {metrics['est']:.4f} &  {metrics['std'] * FACTOR:.4f}  & \\underline{a}{metrics['acc']:.4f}{b}"
                    text_markdown = f"| {metrics['est']:.4f} |  {metrics['std'] * FACTOR:.4f}  | *{metrics['acc']:.4f}*"
                    return text_latex if type == "latex" else text_markdown
                else:
                    text_latex = f"& {metrics['est']:.4f} &  {metrics['std'] * FACTOR:.4f}  & {metrics['acc']:.4f}"
                    text_markdown = f"| {metrics['est']:.4f} |  {metrics['std'] * FACTOR:.4f}  | {metrics['acc']:.4f}"
                    return text_latex if type == "latex" else text_markdown

            def _gen_row(R, type="latex"):
                CMC = results[(R, m1)]["sampling"][g_name]
                PRP_m1 = results[(R, m1)]["proportional"][method][g_name]
                OPT_m1 = results[(R, m1)]["optimal"][method][g_name]
                PRP_m2 = results[(R, m2)]["proportional"][method][g_name]
                OPT_m2 = results[(R, m2)]["optimal"][method][g_name]

                accuracies = [
                    CMC["acc"],
                    PRP_m1["acc"],
                    OPT_m1["acc"],
                    PRP_m2["acc"],
                    OPT_m2["acc"],
                ]
                # Find the best and second best accuracy
                best_acc = max(accuracies)
                if best_acc != 0.0:
                    second_best_acc = max(
                        [acc for acc in accuracies if acc != best_acc]
                    )
                else:
                    second_best_acc = 0.0

                return f"""
                    {_gen_cell(CMC, best_acc, second_best_acc, type=type)} % CMC
                    {_gen_cell(PRP_m1, best_acc, second_best_acc, type=type)} % Prp m 4
                    {_gen_cell(OPT_m1, best_acc, second_best_acc, type=type)} % Opt m 4
                    {_gen_cell(PRP_m2, best_acc, second_best_acc, type=type)} % Prp m 16
                    {_gen_cell(OPT_m2, best_acc, second_best_acc, type=type)} % Opt m 16
                """

            a = "{"
            b = "}"
            if target_value is None:
                target_value = 0.0
            text_latex = f"""
            \\multirow{a}2{b}{a}*{b}{a}{g_name}{b} & \\multirow{a}2{b}{a}*{b}{a}{target_value:.4f}{b} & \\multirow{a}2{b}{a}*{b}{a}{y_obs_est:.4f}{b} & \\multirow{a}2{b}{a}*{b}{a}{y_obs_std * FACTOR:.4f}{b}  & \\multirow{a}2{b}{a}*{b}{a}{y_obs_acc:.4f}{b} &   $2^{a}{log2_R1}{b}$
                {_gen_row(R=R1)}
               \\\\
               &   &   &   &   &   $2^{a}{log2_R2}{b}$
                {_gen_row(R=R2)}
               \\\\ \\hline
            """
            text_markdown = f"""
            | {g_name} | {target_value:.4f} | {y_obs_est:.4f} | {y_obs_std * FACTOR:.4f} | {y_obs_acc:.4f} | $2^{a}{log2_R1}{b}$ |
                {_gen_row(R=R1, type="markdown")}
                |   |   |   |   |   | $2^{a}{log2_R2}{b}$ |
                {_gen_row(R=R2, type="markdown")}
                |   |   |   |   |   |   |
            """

            print(f"g={g_name}, method={method}, R1={R1}, R2={R2}, m1={m1}, m2={m2}")
            if print_latex:
                print(text_latex)
            if print_markdown:
                print(text_markdown)

            if filepath:
                with open(filepath, "a") as f:
                    f.write(
                        f"g={g_name}, method={method}, R1={R1}, R2={R2}, m1={m1}, m2={m2}\n"
                    )
                    f.write(text_latex)
                    f.write("\n")


def estimate_and_print_results(
    scenario: str,
    data_filepath: str,
    dim: int,
    model_filepath: str,
    target_functions: dict,
    target_estimations: dict,
    r1: int,
    r2: int,
    m1: int,
    m2: int,
    repeats: int,
    methods: List[str],
    sample_from_model_using_grid_points: callable,
):
    print(
        f"scenario={scenario}, data_filepath={data_filepath}, model_filepath={model_filepath}, repeats={repeats}"
    )
    results = estimate(
        data_filepath=data_filepath,
        dim=dim,
        model_filepath=model_filepath,
        target_functions=target_functions,
        target_estimations=target_estimations,
        R1=r1,
        R2=r2,
        m1=m1,
        m2=m2,
        methods=methods,
        repeats=repeats,
        sample_from_model_using_grid_points=sample_from_model_using_grid_points,
    )
    # get filename of datapath
    data_filename = data_filepath.split("/")[-1].split(".")[0]
    # get filename of modelpath
    model_filename = model_filepath.split("/")[-1].split(".")[0]

    # save results
    with open(
        f"results___{data_filename}_{model_filename}_{scenario}_{repeats}_{r1}_{r2}_{m1}_{m2}.pkl",
        "wb",
    ) as f:
        pickle.dump(results, f)

    results = average_dict_values(results)
    print_results(
        results=results,
        target_estimations=target_estimations,
        R1=r1,
        R2=r2,
        m1=m1,
        m2=m2,
        print_latex=True,
        print_markdown=True,
        filepath=f"results___{data_filename}_{model_filename}_{scenario}_{repeats}_{r1}_{r2}_{m1}_{m2}.md",
        methods=methods,
    )


def parse_arguments(command_args: list[str]):
    """
    Parse command line arguments for CNF_lib models.
    Args:
        command_args: list of command line arguments.
        Example:
            command_args = [
                "--n_samples",
                "1000",
                "--n_epochs",
                "100",
            ]
    Returns:
        parsed arguments.
    """
    NONLINEARITIES = ["tanh", "relu", "softplus", "elu", "swish", "square", "identity"]
    SOLVERS = [
        "dopri5",
        "bdf",
        "rk4",
        "midpoint",
        "adams",
        "explicit_adams",
        "fixed_adams",
    ]
    LAYERS = [
        "ignore",
        "concat",
        "concat_v2",
        "squash",
        "concatsquash",
        "concatcoord",
        "hyper",
        "blend",
    ]

    import argparse

    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument(
        "--layer_type",
        type=str,
        default="concatsquash",
        choices=LAYERS,
    )
    parser.add_argument("--dims", type=str, default="16-16")
    parser.add_argument(
        "--num_blocks", type=int, default=2, help="Number of stacked CNFs."
    )
    parser.add_argument("--time_length", type=float, default=0.5)
    parser.add_argument("--train_T", type=eval, default=True)

    parser.add_argument(
        "--n_mix", type=int, default=2, help="only for GMMHMM: number of mixtures"
    )

    parser.add_argument("--add_noise", type=eval, default=False, choices=[True, False])
    parser.add_argument("--noise_var", type=float, default=0.1)
    parser.add_argument(
        "--divergence_fn",
        type=str,
        default="brute_force",
        choices=["brute_force", "approximate"],
    )
    parser.add_argument(
        "--nonlinearity", type=str, default="tanh", choices=NONLINEARITIES
    )

    parser.add_argument("--solver", type=str, default="dopri5", choices=SOLVERS)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--step_size", type=float, default=None, help="Optional fixed step size."
    )

    parser.add_argument(
        "--test_solver", type=str, default=None, choices=SOLVERS + [None]
    )
    parser.add_argument("--test_atol", type=float, default=None)
    parser.add_argument("--test_rtol", type=float, default=None)

    parser.add_argument("--residual", type=eval, default=False, choices=[True, False])
    parser.add_argument("--rademacher", type=eval, default=False, choices=[True, False])
    parser.add_argument(
        "--spectral_norm", type=eval, default=False, choices=[True, False]
    )
    parser.add_argument("--batch_norm", type=eval, default=True, choices=[True, False])
    parser.add_argument("--bn_lag", type=float, default=0)
    args = parser.parse_args(command_args)
    LOGGER.info(args)
    return args
