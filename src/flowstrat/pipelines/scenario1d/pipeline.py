from kedro.pipeline import Pipeline

from ..node_utils import create_general_pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return create_general_pipeline(scenario_name="scenario1d")
