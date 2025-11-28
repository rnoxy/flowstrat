"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline

import flowstrat.pipelines.sanity_check
import flowstrat.pipelines.scenario1d
import flowstrat.pipelines.scenario2d
import flowstrat.pipelines.scenarioTS3d


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    pipelines = {
        "sanity_check": flowstrat.pipelines.sanity_check.create_pipeline(),
        "scenario1d": flowstrat.pipelines.scenario1d.create_pipeline(),
        "scenario2d": flowstrat.pipelines.scenario2d.create_pipeline_2d(),
        "scenarioWind2d": flowstrat.pipelines.scenario2d.create_pipeline_wind2d(),
        "scenarioTS3d": flowstrat.pipelines.scenarioTS3d.create_pipeline(),
    }

    pipelines["__default__"] = sum(pipelines.values(), Pipeline([]))
    return pipelines
