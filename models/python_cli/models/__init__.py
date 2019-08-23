from typing import Dict, List, Type

from common import ModelSpec

from .iris import IrisModel
from .mnist_cnn import MnistCnnModel
from .mnist_mlp import MnistMlpModel


_model_spec_list: List[Type[ModelSpec]] = [
    IrisModel,
    MnistCnnModel,
    MnistMlpModel,
]


model_specs: Dict[str, Type[ModelSpec]] = {ms.name: ms for ms in _model_spec_list}
