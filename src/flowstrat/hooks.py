from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline

from flowstrat.utils import set_seed_everywhere


class ProjectHooks:
    @hook_impl
    def before_pipeline_run(
        self, run_params: dict[str, any], pipeline: Pipeline, catalog: DataCatalog
    ) -> None:
        pass
