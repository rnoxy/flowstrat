import kedro.io
import numpy as np
import torch
from kedro.pipeline import Pipeline

from flowstrat.pipelines.node_utils import (
    create_model_node,
    train_model_node,
    evaluate_model_node,
)
from flowstrat.pipelines.scenario1d import create_pipeline
from flowstrat.pipelines.scenario1d.nodes import (
    sample_data_1D,
    get_target_functions_and_estimations,
    sample_from_flow_model_using_1d_grid_points,
)
from flowstrat.utils import parse_arguments, set_seed_everywhere


def test_create_pipeline():
    pipeline = create_pipeline()
    assert isinstance(pipeline, Pipeline)


def test_dataset_creation_node():
    dataset = sample_data_1D(n_samples=2000)
    assert isinstance(dataset, np.ndarray)
    assert dataset.shape == (2000, 1)


def _all_close_values(a, b):
    atol = 1e-3
    if isinstance(a, dict):
        assert isinstance(b, dict)
        for key in a.keys():
            assert key in b.keys()
            _all_close_values(a[key], b[key])
    elif isinstance(a, list):
        assert isinstance(b, list)
        assert len(a) == len(b)
        for i in range(len(a)):
            _all_close_values(a[i], b[i])
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray)
        assert a.shape == b.shape
        assert np.allclose(a, b, atol=atol)
    elif isinstance(a, float):
        assert isinstance(b, float)
        assert np.allclose(a, b, atol=atol)
    elif isinstance(a, np.float32):
        assert isinstance(b, np.float32)
        assert np.allclose(a, b, atol=atol)
    elif isinstance(a, int):
        assert isinstance(b, int)
        assert a == b
    elif isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor)
        assert a.shape == b.shape
        assert torch.allclose(a, b, atol=atol)
    else:
        raise ValueError(f"Unknown type {type(a)}")


def test_pipeline(catalog: kedro.io.DataCatalog):
    device = "cpu"
    n_samples = 1000
    n_epochs = 20
    lrate = 0.001

    set_seed_everywhere(42)
    model_params = catalog.load("params:scenario1d.model")
    dataset_train = sample_data_1D(n_samples=n_samples)
    model_args = parse_arguments(model_params["args"])
    model = create_model_node(
        args=model_args,
        device=device,
        input_dim=model_params["input_dim"],
    )
    train_model_node(
        model=model,
        samples=dataset_train,
        params=dict(
            n_epochs=n_epochs,
            lrate=lrate,
            weight_decay=0.00001,
        ),
        device=device,
    )

    evaluation_results = evaluate_model_node(
        model=model,
        params=catalog.load("params:scenario1d.model_evaluation"),
        training_samples=dataset_train,
        device=device,
        scenario_name="scenario1d",
    )

    # load expected results from "data/scenario1d/test.pkl"
    import pickle

    expected_results = pickle.load(open("data/scenario1d/test.pkl", "rb"))

    _all_close_values(evaluation_results, expected_results)
