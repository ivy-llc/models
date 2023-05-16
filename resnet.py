from torchvision.models import resnet18, ResNet18_Weights
from ivy_models.resnet import resnet_18, ResNet, ResidualBlock
import torch
import ivy

path = "ivy_models/resnet/pretrained_weights/resnet_18.pickled"

# torch_resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
# torch_w = ivy.Container(torch_resnet.state_dict())
# torch_w.cont_to_disk_as_pickled(path)
# print(torch_resnet)

# torch_resnet = resnet18()
# torch_resnet.load_state_dict(torch.load(path))
# print(torch_resnet)
# print(torch_resnet.state_dict())
# print(ivy.Container(torch_resnet.state_dict()))


ivy.set_backend("torch")
ivy_v = ivy.Container.cont_from_disk_as_pickled(path)
# resnet = ResNet(ResidualBlock, )
resnet = resnet_18(v=ivy_v)
print(resnet)