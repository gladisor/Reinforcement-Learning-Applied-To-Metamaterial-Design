import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class PPOModel(TorchModelV2, nn.Module):