# inspired heavily by pytorch's efficient -
# https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py

import ivy
import ivy_models
from ivy_models.inceptionnet.layers import (BasicConv2d,)
from ivy_models.base import BaseSpec, BaseModel
import builtins
from typing import Callable, Optional, Sequence, Union, Tuple, List


