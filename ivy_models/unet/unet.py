import ivy


def double_conv(in_c, out_c):
    conv = ivy.Sequential(
        ivy.Conv2D(in_c, out_c, [3, 3], 1, 0),
        ivy.ReLU(),
        ivy.Conv2D(out_c, out_c, [3, 3], 1, 0),
        ivy.ReLU(),
    )
    return conv


def crop_img(tensor, target_tensor):
    target_size = target_tensor.shape[2]
    tensor_size = tensor.shape[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, delta : tensor_size - delta, delta : tensor_size - delta, :]


class UNet(ivy.Module):
    def __init__(self, in_channel=1, out_channel=2, v=None):
        super(UNet, self).__init__()
        if v is not None:
            self.v = v
        self.pool = ivy.MaxPool2D(2, 2, 0)
        self.down_conv_1 = double_conv(in_channel, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)
        self.up_trans_1 = ivy.Conv2DTranspose(1024, 512, [2, 2], 2, "VALID")
        self.up_conv_1 = double_conv(1024, 512)
        self.up_trans_2 = ivy.Conv2DTranspose(512, 256, [2, 2], 2, "VALID")
        self.up_conv_2 = double_conv(512, 256)
        self.up_trans_3 = ivy.Conv2DTranspose(256, 128, [2, 2], 2, "VALID")
        self.up_conv_3 = double_conv(256, 128)
        self.up_trans_4 = ivy.Conv2DTranspose(128, 64, [2, 2], 2, "VALID")
        self.up_conv_4 = double_conv(128, 64)
        self.out = ivy.Conv2D(64, out_channel, [1, 1], 1, 0)

    def _forward(self, image):
        # B, H, W, C
        # encoder
        x1 = self.down_conv_1(image)
        x2 = self.pool(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.pool(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.pool(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.pool(x7)
        x9 = self.down_conv_5(x8)

        # decoder
        x = self.up_trans_1(x9)
        y = crop_img(x7, x)
        x = self.up_conv_1(ivy.concat([x, y], axis=-1))
        x = self.up_trans_2(x)
        y = crop_img(x5, x)
        x = self.up_conv_2(ivy.concat([x, y], axis=-1))
        x = self.up_trans_3(x)
        y = crop_img(x3, x)
        x = self.up_conv_3(ivy.concat([x, y], axis=-1))
        x = self.up_trans_4(x)
        y = crop_img(x1, x)
        x = self.up_conv_4(ivy.concat([x, y], axis=-1))
        x = self.out(x)
        return x


if __name__ == "__main__":
    ivy.set_torch_backend()
    image = ivy.random_normal(shape=(1, 572, 572, 1))
    model = UNet()
    model(image)
