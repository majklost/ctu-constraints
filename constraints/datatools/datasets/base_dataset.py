from abc import ABC, abstractmethod

import torch

from ..label_schema import LabelSchema
from .types import Sample


class BaseDataset(torch.utils.data.Dataset, ABC):
    """Abstract base class for datasets.

    This class defines the interface that all dataset classes should implement.
    Subclasses must provide implementations for the `__len__` and `__getitem__`
    methods to allow for proper indexing and iteration over the dataset samples.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Sample:
        """Return a sample from the dataset at the given index."""
        pass

    @property
    @abstractmethod
    def label_schema(self) -> LabelSchema:
        """Return the label schema for the dataset."""
        pass
