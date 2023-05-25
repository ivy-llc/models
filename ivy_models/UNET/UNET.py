import ivy
ivy.set_backend('tensorflow')

def double_conv(in_c, out_c):
    conv = ivy.Sequential(
        ivy.Conv2D(in_c, out_c, [3, 3], 1, 0),
        ivy.ReLU(),
        ivy.Conv2D(out_c, out_c, [3, 3], 1, 0),
        ivy.ReLU(),
    )
    return conv

class UNet(ivy.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.pool = ivy.MaxPool2D(2, 2, 0)
        self.down_conv_1 = double_conv(1, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

    def _forward(self, image):
        x1 = self.down_conv_1(image)
        print(x1.shape)
        x2 = self.pool(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.pool(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.pool(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.pool(x7)
        x9 = self.down_conv_5(x8)
        print(x9.shape)

image = ivy.random_normal(shape=(1, 572, 572, 1))
model = UNet()
print(model(image))