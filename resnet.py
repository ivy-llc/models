from torchvision.models import resnet18, ResNet18_Weights
from ivy_models.resnet import resnet_18, ResNet, ResidualBlock
from ivy_models.transformers import PerceiverIO, PerceiverIOSpec
import torch
import ivy

path = "ivy_models/resnet/pretrained_weights/resnet_18.pickled"

# torch save weights

torch_resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
# print(torch_resnet([1]))
# torch_w = ivy.Container(torch_resnet.state_dict())
# torch_w.cont_to_disk_as_pickled(path)
# print(torch_resnet)





# torch load weights

# torch_resnet = resnet18()
# torch_resnet.load_state_dict(torch.load(path))
# print(torch_resnet)
# print(torch_resnet.state_dict())
# print(ivy.Container(torch_resnet.state_dict()))





# perceiver .v attribute test

perceiver = PerceiverIO(PerceiverIOSpec(3,2,1000, learn_query=False))
print(perceiver.v)







# ivy load weights

# ivy.set_backend("torch")
ivy_v = ivy.Container.cont_from_disk_as_pickled(path)
# # resnet = ResNet(ResidualBlock, )
resnet = resnet_18(v=ivy_v)
# resnet = resnet_18()
# print(resnet.v)
# print(resnet([1]))