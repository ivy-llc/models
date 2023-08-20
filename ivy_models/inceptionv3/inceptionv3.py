# global
import ivy
import ivy_models
from ivy_models.base import BaseSpec, BaseModel
from .layers import ConvBlock, Inception, Auxiliary


class InceptionV3Spec(BaseSpec):
    def __init__(
        self, num_classes=1000, dropout=0.4, aux_dropout=0.7, data_format="NCHW"
    ):
        super(InceptionV3Spec, self).__init__(
            num_classes=num_classes,
            dropout=dropout,
            aux_dropout=aux_dropout,
            data_format=data_format,
        )

    
class InceptionV3(BaseModel):
    """
    Inception-V3 architecture.

    Args::
        num_classes (int): Number of output classes. Defaults to 1000.
        v (ivy.Container): Unused parameter. Can be ignored.

    """
    
    def __init__(
        self,
        num_classes= 1000,
        training_mode = False,
        aux_logits= True,
        dropout= 0.5,
        data_format="NCHW",
        spec=None,
        v=None
        ):
            self.spec = (
                spec
                if spec and isinstance(spec, InceptionV3Spec)
                else InceptionV3Spec(
                    num_classes=num_classes,
                    dropout=dropout,
                    aux_dropout=dropout,
                    aux_logits=aux_logits,
                    training=training_mode,
                    data_format=data_format,
                )
            )
            super(InceptionV3, self).__init__(v=self.v)
            
        
    def _build(self, *args, **kwargs):
        # blocks
        self.conv_block = BasicConv2d
        self.inception_a = InceptionA
        self.inception_b = InceptionB
        self.inception_c = InceptionC
        self.inception_d = InceptionD
        self.inception_e = InceptionE
        self.inception_aux = InceptionAux
        
        self.Conv2d_1a_3x3 = self.conv_block(3, 32, kernel_size=[3,3], stride=2)
        self.Conv2d_2a_3x3 = self.conv_block(32, 32, kernel_size=[3,3])
        self.Conv2d_2b_3x3 = self.conv_block(32, 64, kernel_size=[3,3], padding=[[1,1],[1,1]])
        self.maxpool1 = ivy.MaxPool2D([3,3], 2, 0)
        self.Conv2d_3b_1x1 = self.conv_block(64, 80, kernel_size=[1,1])
        self.Conv2d_4a_3x3 = self.conv_block(80, 192, kernel_size=[3,3])
        self.maxpool2 = ivy.MaxPool2D([3,3], 2, 0)
        self.Mixed_5b = self.inception_a(192, pool_features=32)
        self.Mixed_5c = self.inception_a(256, pool_features=64)
        self.Mixed_5d = self.inception_a(288, pool_features=64)
        self.Mixed_6a = self.inception_b(288)
        self.Mixed_6b = self.inception_c(768, channels_7x7=128)
        self.Mixed_6c = self.inception_c(768, channels_7x7=160)
        self.Mixed_6d = self.inception_c(768, channels_7x7=160)
        self.Mixed_6e = self.inception_c(768, channels_7x7=192)

        # it is used only when the model is in training mode
        AuxLogits = None
        if self.aux_logits:
            self.AuxLogits = self.inception_aux(768, self.spec.num_classes)

        self.Mixed_7a = self.inception_d(768)
        self.Mixed_7b = self.inception_e(1280)
        self.Mixed_7c = self.inception_e(2048)
        self.avgpool = ivy.AdaptiveAvgPool2d((1, 1))
        self.dropout = ivy.Dropout(prob=self.spec.dropout)
        self.fc = ivy.Linear(2048, self.spec.num_classes)


    @classmethod
    def get_spec_class(self):
        return InceptionV3Spec
    
    def _forward(self, x, data_format=None):
        data_format = data_format if data_format else self.spec.data_format
        if data_format == "NHWC":
            x = ivy.permute_dims(x, (0, 3, 1, 2))
            
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147

        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147

        x = self.maxpool1(x)
        # N x 64 x 73 x 73

        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73

        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71

        x = self.maxpool2(x)
        # N x 192 x 35 x 35

        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35


        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35

        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35

        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17

        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17

        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17

        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17

        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        
        aux = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
                # N x 768 x 17 x 17

        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8

        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8

        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8

        # Adaptive average pooling
        x = ivy.permute_dims(x, (0, 3, 1, 2))
        x = self.avgpool(x)
        x = ivy.permute_dims(x, (0, 2, 3, 1))
        # N x 2048 x 1 x 1

        x = self.dropout(x)
        # N x 2048 x 1 x 1

        x = ivy.flatten(x, start_dim=1)
        # N x 2048

        x = self.fc(x)
        # N x 1000 (num_classes)

        return x, aux
    

def _inceptionNet_v3_torch_weights_mapping(old_key, new_key):
    W_KEY = ["conv/weight"]
    new_mapping = new_key
    if any([kc in old_key for kc in W_KEY]):
        new_mapping = {"key_chain": new_key, "pattern": "b c h w -> h w c b"}
    return new_mapping


def ivy_inceptionNet_v3(pretrained=True, num_classes=1000, dropout=0, data_format="NHWC"):
    """InceptionNet-V3 model"""
    model = InceptionV3(pretrained=True, num_classes=1000, dropout=0, data_format="NHWC")
    if pretrained:
        url = "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked", "AuxLogits"],
            custom_mapping=_inceptionNet_v3_torch_weights_mapping,
        )
        model.v = w_clean
    return model

