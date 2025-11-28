from kedro.pipeline import Pipeline, node, pipeline
from .nodes import sanity_check


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=sanity_check,
                inputs="parameters",
                outputs="sanity_check_output",
                name="sanity_check",
            ),
        ]
    )
