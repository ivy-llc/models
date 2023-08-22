# inspired heavily by pytorch's efficient -
# https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py

import ivy
import ivy_models
from ivy_models.inceptionnet.layers import (BasicConv2d, InceptionAux, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE)
from ivy_models.base import BaseSpec, BaseModel
import builtins
from typing import Callable, Optional, Sequence, Union, Tuple, List

# log sys
import sys
sys.path.append("/ivy_models/log_sys/pf.py")
from log_sys.pf import *
# log sys

class InceptionV3(ivy.Module):
    def __init__(
        self,
        num_classes: int=1000,
        dropout: float=0.5,
        data_format="NHWC",
        ) -> None:
        self.num_classes = num_classes
        self.dropout = dropout
        self.data_format=data_format
        super().__init__()


    def _build(self, *args, **kwargs):
        conv_block = BasicConv2d
        inception_a = InceptionA
        inception_b = InceptionB
        inception_c = InceptionC
        inception_d = InceptionD
        inception_e = InceptionE
        inception_aux = InceptionAux

        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=[3,3], stride=2)
        pf(f"InceptionV3 | build | layer 1/22 built")
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=[3,3])
        pf(f"InceptionV3 | build | layer 2/22 built")
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=[3,3], padding=[[1,1],[1,1]])
        pf(f"layer 3/22 built")
        self.maxpool1 = ivy.MaxPool2D([3,3], 2, 0)
        pf(f"InceptionV3 | build | layer 4/22 built")
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=[1,1])
        pf(f"InceptionV3 | build | layer 5/22 built")
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=[3,3])
        pf(f"InceptionV3 | build | layer 6/22 built")
        self.maxpool2 = ivy.MaxPool2D([3,3], 2, 0)
        pf(f"InceptionV3 | build | layer 7/22 built")
        self.Mixed_5b = inception_a(192, pool_features=32)
        pf(f"InceptionV3 | build | layer 8/22 built")
        self.Mixed_5c = inception_a(256, pool_features=64)
        pf(f"InceptionV3 | build | layer 9/22 built")
        self.Mixed_5d = inception_a(288, pool_features=64)
        pf(f"InceptionV3 | build | layer 10/22 built")
        self.Mixed_6a = inception_b(288)
        pf(f"InceptionV3 | build | layer 11/22 built")
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        pf(f"InceptionV3 | build | layer 12/22 built")
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        pf(f"InceptionV3 | build | layer 13/22 built")
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        pf(f"InceptionV3 | build | layer 14/22 built")
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        pf(f"InceptionV3 | build | layer 15/22 built")

        # if is used only when the model is in training mode
#         self.AuxLogits = inception_aux(768, num_classes)
#         pf(f"layer 16/22 built")

        self.Mixed_7a = inception_d(768)
        pf(f"InceptionV3 | build | layer 17/22 built")
        self.Mixed_7b = inception_e(1280)
        pf(f"InceptionV3 | build | layer 18/22 built")
        self.Mixed_7c = inception_e(2048)
        pf(f"InceptionV3 | build | layer 19/22 built")
        self.avgpool = ivy.AdaptiveAvgPool2d((1, 1))
        pf(f"InceptionV3 | build | layer 20/22 built")
        self.dropout = ivy.Dropout(prob=self.dropout)
        pf(f"InceptionV3 | build | layer 21/22 built")
        self.fc = ivy.Linear(2048, self.num_classes)
        pf(f"InceptionV3 | build | layer 22/22 built")


    def _forward(self, x: ivy.Array) -> List[ivy.Array]:
        pf(f"InceptionV3 | forward | input shape is:{x.shape} | done 1/27")

        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 2/27")

        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 3/27")

        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 4/27")

        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 5/27")

        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        pf(f"InceptionV3 | forward | is:{x.shape} | done 6/27")

        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 7/27")

        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 8/27")

        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
#         pf(f"v3 8/27, output type is:{type(x)}")
#         pf(f"v3 8/27, output len is:{len(x)}")
        pf(f"InceptionV3 | forward |shape is:{x.shape} | done 9/27")


        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 10/27")

        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 11/27")

        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 12/27")

        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        pf(f"InceptionV3 | forward |shape is:{x.shape} | done 13/27")

        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 14/27")

        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 15/27")

        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 16/27")

#         aux = self.AuxLogits(x)
#         # N x 768 x 17 x 17
#         pf(f"v3 16/27, output shape is:{x.shape}")

        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 17/27")

        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 18/27")

        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 19/27")

        # Adaptive average pooling
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 20/27")
        x = ivy.permute_dims(x, (0, 3, 1, 2))
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 21/27")
        x = self.avgpool(x)
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 22/27")
        x = ivy.permute_dims(x, (0, 2, 3, 1))
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 27/27")

        # N x 2048 x 1 x 1
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 24/27")

        x = self.dropout(x)
        # N x 2048 x 1 x 1
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 25/27")

        x = ivy.flatten(x, start_dim=1)
        # N x 2048
        pf(f"InceptionV3 | forward | shape is:{x.shape} | done 26/27")

        x = self.fc(x)
        # N x 1000 (num_classes)
        pf(f"InceptionV3 | forward | fc | done 27/27")

#         return x, aux
        return x


def _inceptionNet_v3_torch_weights_mapping(old_key, new_key):
    W_KEY = ["conv/weight"]
    new_mapping = new_key
    if any([kc in old_key for kc in W_KEY]):
        new_mapping = {"key_chain": new_key, "pattern": "b c h w -> h w c b"}
    return new_mapping


def inceptionNet_v3(pretrained=True, num_classes=1000, dropout=0.5, data_format="NHWC"):
    """InceptionNet-V3 model"""

    model = InceptionV3(num_classes=num_classes, dropout=dropout, data_format=data_format)
    # pf(f"my model weights are:");pprint.pprint(ivy.Container(model.v))
    pf("inceptionNet_v3 | building InceptionV3 model | done 1/3")
    if pretrained:
        url = "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked", "AuxLogits"],
            custom_mapping=_inceptionNet_v3_torch_weights_mapping,
        )
        pf("inceptionNet_v3 | clearning pretrained weights | done 2/3")

        model.v = w_clean
        pf("inceptionNet_v3 | loading pretrained weights | done 3/3")
    return model