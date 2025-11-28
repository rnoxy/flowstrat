from kedro.pipeline import Pipeline

from ..node_utils import create_general_pipeline


def create_pipeline_2d(**kwargs) -> Pipeline:
    return create_general_pipeline(scenario_name="scenario2d")


def create_pipeline_wind2d(**kwargs) -> Pipeline:
    return create_general_pipeline(scenario_name="scenarioWind2d")
