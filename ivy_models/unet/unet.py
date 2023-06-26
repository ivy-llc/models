from ivy_models.helpers import load_torch_weights
import ivy
from unet_parts import *
import builtins


class UNET(ivy.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, v=None):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        super(UNET, self).__init__(v=v)

    def _forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
def _unet_torch_weights_mapping(old_key, new_key):
    new_mapping = new_key
    W_KEY = [
        "0/weight",
        "3/weight",
        "conv/weight",
    ]

    if "up/weight" in old_key:
        new_mapping = {"key_chain": new_key, "pattern": "b c h w -> h w b c"}
    elif builtins.any([kc in old_key for kc in W_KEY]):
        new_mapping = {"key_chain": new_key, "pattern": "b c h w -> w h c b"}
    elif "conv/bias" in old_key or "up/bias" in old_key:
        new_mapping = {"key_chain": new_key, "pattern": "h -> 1 1 1 h"}
    
    return new_mapping


def UNet(n_channels=3, n_classes=1, v=None, pretrained=False):
    """UNET model"""
    if not pretrained:
        return UNET(n_channels=n_channels, n_classes=n_classes, v=v)

    reference_model = UNET(n_channels=n_channels, n_classes=n_classes)
    url = "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth"
    w_clean = load_torch_weights(
        url, reference_model, custom_mapping=_unet_torch_weights_mapping
    )
    return UNET(n_channels=n_channels, n_classes=n_classes, v=w_clean)

def preprocess(mask_values, pil_img, scale, is_mask):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
    img = np.asarray(pil_img)

    if is_mask:
        mask = np.zeros((newH, newW), dtype=np.int64)
        for i, v in enumerate(mask_values):
            if img.ndim == 2:
                mask[img == v] = i
            else:
                mask[(img == v).all(-1)] = i

        return mask

    else:
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))

        if (img > 1).any():
            img = img / 255.0

        return img
if __name__ == "__main__":
    # Preprocess torch image
    from torchvision import transforms
    from PIL import Image
    import torch
    import numpy as np
    ivy.set_torch_backend()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --------------- TO REMOVE ------------------
    # print(device)
    # filename = '/workspaces/models/ivy_models/unet/Screenshot 2023-06-24 091018.png'
    # torch_img = Image.open(filename)
    # # Convert the image to RGB if it has an alpha channel
    # if torch_img.mode != 'RGB':
    #     torch_img = torch_img.convert('RGB')
    # np.save('resent_image', torch_img)
    # --------------- TO REMOVE ------------------

    # Loading data
    np_array = np.load('ivy_models/unet/resent_image.npy')
    # prepaaring data
    pil_image = Image.fromarray(np_array)
    torch_img = torch.from_numpy(preprocess(None, pil_image, 0.5, is_mask=False))
    torch_img = torch_img.unsqueeze(0)
    torch_img = torch_img.to(device=device, dtype=torch.float32)
    torch_img = torch_img.numpy().reshape(1, torch_img.shape[2], torch_img.shape[3], 3)

    # print(torch_img.shape)
    # defining model with pretrained weights
    net = UNet(n_channels=3, n_classes=2, pretrained=True)
    
    # predicting
    # with torch.no_grad():
    #     output = net(torch_img).cpu()
    #     output = F.interpolate(output, (pil_image.size[1], pil_image.size[0]), mode='bilinear')
    #     if net.n_classes > 1:
    #         mask = output.argmax(dim=1)
    #     else:
    #         mask = torch.sigmoid(output) > out_threshold

    # mask = mask[0].long().squeeze().numpy()

    # ----------- TO REMOVE ----------
    # # print(t)
    # pred = net(torch_img)
    # print(pred)





    # import torch
    # import numpy as np
    # import os

    # # Load image
    # this_dir = os.path.dirname(os.path.realpath(__file__))
    # device = "cpu"

    # img = ivy.asarray(
    #     np.load(os.path.join(this_dir, "new_array.npy"))[None],
    #     dtype="float32",
    #     device=device,
    # )
    # # img = ivy.expand_dims(img, axis=-1)
    # print(img[0].shape)

    # ivy.set_torch_backend()
    # net = UNet(n_channels=3, n_classes=2, pretrained=True)

    # # print(reference_model.v.pop('mask_values',[0,1]))

    # mask_values = net.v.pop('mask_values',[0,1])

    # pred = net(img[0])
    # print(pred)

    # print(ivy.Container(reference_model.v))
