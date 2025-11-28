from kedro.pipeline import Pipeline, pipeline, node

from .nodes import generate_dataset, train_model_and_split, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        pipe=[
            node(
                func=generate_dataset,
                inputs="params:dataset_generation",
                outputs=["dataset_train", "dataset_test"],
                name="generate_dataset",
            ),
            node(
                func=train_model_and_split,
                inputs=["dataset_train", "params:model_training"],
                outputs=["model", "history"],
                name="train_model",
            ),
            node(
                func=evaluate_model,
                inputs=["dataset_train", "model", "params:model_evaluation"],
                outputs="evaluation_results",
                name="evaluate_model",
            ),
        ],
        namespace="scenarioTS3d",
    )
