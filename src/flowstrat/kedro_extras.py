import torch
from kedro.io import AbstractDataSet
import logging


class TorchModelDataSet(AbstractDataSet):
    def __init__(self, filepath: str):
        self._filepath = filepath
        super().__init__()

    def _load(self) -> torch.nn.Module:
        return torch.load(self._filepath)

    def _save(self, model: torch.nn.Module) -> None:
        self._logger.debug(f"Saving model to {self._filepath}")
        torch.save(model, self._filepath)

    def _describe(self):
        return "TorchModelDataSet"
