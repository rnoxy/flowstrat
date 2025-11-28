import pprint

import numpy as np
import torch
import torch.nn
from kedro.pipeline import pipeline, node
from tqdm import tqdm

from CNF_lib.layers import SequentialFlow
from flowstrat.cnf_utils import build_model_tabular, standard_normal_logprob
from flowstrat.utils import (
    set_seed_everywhere,
    calculate_results,
    average_dict_values,
    parse_arguments,
)

import logging

LOGGER = logging.getLogger(__name__)


def create_general_pipeline(scenario_name: str):
    return pipeline(
        pipe=[
            node(
                func=generate_dataset_node,
                inputs={
                    "scenario_name": "params:name",
                    "params": "params:dataset_generation",
                },
                outputs="dataset_train",
                name="generate_dataset",
            ),
            node(
                func=parse_arguments,
                inputs="params:model.args",
                outputs="model_args",
                name="parse_arguments",
            ),
            node(
                func=create_model_node,
                inputs={
                    "args": "model_args",
                    "device": "params:device",
                    "input_dim": "params:model.input_dim",
                },
                outputs="initial_model",
                name="create_model",
            ),
            node(
                func=train_model_node,
                inputs={
                    "model": "initial_model",
                    "samples": "dataset_train",
                    "params": "params:model_training",
                    "device": "params:device",
                },
                outputs="model",
                name="train_model",
            ),
            node(
                func=evaluate_model_node,
                inputs={
                    "model": "model",
                    "training_samples": "dataset_train",
                    "device": "params:device",
                    "params": "params:model_evaluation",
                    "scenario_name": "params:name",
                },
                outputs="evaluation_results",
                name="evaluate_model",
            ),
            node(
                func=present_results_node,
                inputs={
                    "results": "evaluation_results",
                },
                outputs=None,
                name="print_results",
            ),
        ],
        namespace=scenario_name,
        parameters={
            "device": "params:device",
        },
    )


def generate_dataset_node(scenario_name: str, params: dict):
    n_samples = params["n_train"]
    if scenario_name == "scenario1d":
        from flowstrat.pipelines.scenario1d.nodes import sample_data_1D

        return sample_data_1D(n_samples=n_samples)
    elif scenario_name == "scenario2d":
        from flowstrat.pipelines.scenario2d.nodes import sample_data_2D

        return sample_data_2D(n_samples=n_samples)
    elif scenario_name == "scenarioWind2d":
        import pandas as pd

        df = pd.read_csv(params["filepath"])
        LOGGER.info(f"Loaded data from {params['filepath']}")
        df_data = df[params["columns"]].values[:n_samples]
        LOGGER.info(f"Data shape: {df_data.shape}")
        return df_data
    elif scenario_name == "scenarioTS3d":
        from flowstrat.pipelines.scenarioTS3d.nodes import generate_dataset

        return generate_dataset(params=params)
    else:
        raise ValueError(f"Unknown scenario {scenario_name}")


def create_model_node(args, device: str, input_dim: int) -> torch.nn.Module:
    """Create the model."""
    return build_model_tabular(args=args, dims=input_dim).to(device)


def train_cnf_model(
    cnf: SequentialFlow,
    data: torch.Tensor,
    n_epochs: int,
    lrate: float,
    weight_decay: float,
):
    optimizer = torch.optim.Adam(cnf.parameters(), lr=lrate, weight_decay=weight_decay)

    pbar = tqdm(range(n_epochs))
    for _ in pbar:
        optimizer.zero_grad()
        y, delta_log_py = cnf(data, torch.zeros(data.size(0), 1).to(data))
        log_py = standard_normal_logprob(y).sum(1)
        delta_log_py = delta_log_py.sum(1)
        log_px = log_py - delta_log_py
        log_px = -log_px.mean()
        log_px.backward()
        optimizer.step()
        pbar.set_postfix(loss=log_px.item())


def train_model_node(
    model: SequentialFlow,
    samples: np.ndarray,
    params: dict[str, any],
    device: str,
):
    samples = torch.tensor(samples).float().to(device)
    train_cnf_model(
        model,
        samples,
        n_epochs=params["n_epochs"],
        lrate=params["lrate"],
        weight_decay=params["weight_decay"],
    )
    return model


def evaluate_model_node(
    model: SequentialFlow,
    training_samples: np.ndarray,
    device: str,
    params: dict,
    scenario_name: str,
):
    from flowstrat.pipelines.scenario1d.nodes import (
        get_target_functions_and_estimations as get1d,
    )
    from flowstrat.pipelines.scenario1d.nodes import (
        sample_from_flow_model_using_1d_grid_points as sample1d,
    )
    from flowstrat.pipelines.scenario2d.nodes import (
        get_target_functions_and_estimations as get2d,
    )
    from flowstrat.pipelines.scenario2d.nodes import (
        get_target_functions_and_estimations_for_wind2d as get_wind2d,
    )
    from flowstrat.pipelines.scenario2d.nodes import (
        sample_from_flow_model_using_2d_grid_points as sample2d,
    )

    set_seed_everywhere(params.get("seed", 42))

    if scenario_name == "scenario1d":
        target_functions, target_estimations = get1d()
        sample_func = sample1d
    elif scenario_name == "scenario2d":
        target_functions, target_estimations = get2d()
        sample_func = sample2d
    elif scenario_name == "scenarioWind2d":
        target_functions, target_estimations = get_wind2d()
        sample_func = sample2d
    else:
        raise ValueError(f"Unknown scenario {scenario_name}")

    methods = params["methods"]
    repeats = params["repeats"]

    model = model.to(device)
    model.eval()

    results = []
    for _ in tqdm(range(repeats)):
        results.append(
            {
                (n, m): calculate_results(
                    data_train=training_samples,
                    model=model,
                    n_samples=n,
                    R=n,
                    m=m,
                    target_functions=target_functions,
                    target_estimations=target_estimations,
                    _methods=methods,
                    sample_from_model_using_grid_points=sample_func,
                    dim=training_samples.shape[1],
                )
                for n in params["n_samples"]
                for m in params["strata"]
            }
        )
    return results


def present_results_node(results: dict) -> None:
    """Present the results."""
    results = average_dict_values(results)

    for (R, m), r in results.items():
        print("-" * 80)
        print(f"n_samples (R) = {R}, strata (m) = {m}")
        r_training = r["training"]
        print("\nY^{obs} - estimations for training data")
        pprint.pprint(r_training)

        r_CMC = r["sampling"]
        print("\nY^{CMC} - estimation for data sampled from the flow model")
        pprint.pprint(r_CMC)

        for method in r["proportional"].keys():
            print("\n***\nmethod:", method)
            r_PROP = r["proportional"][method]
            print("\nY^{PROP} - proportional stratified sampling")
            pprint.pprint(r_PROP)

            r_PROP = r["optimal"][method]
            print("\nY^{OPT} - optimal stratified sampling")
            pprint.pprint(r_PROP)
