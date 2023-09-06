import ivy
import math

class Conv(ivy.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=[1,1], s=1, p=None):
        if isinstance(k, int):  # changing it to a list if int
          k = [k, k]
        # ivy.conv2d does not have groups parameter

        self.conv = ivy.Conv2D(c1, c2, k, s, autopad(k, p), with_bias=False, data_format = "NCHW")
        self.bn = ivy.BatchNorm2D(c2, data_format = "NCHW")
        self.act = ivy.SiLU() 
        super(Conv, self).__init__()

    def _forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class MP(ivy.Module):
    def __init__(self, k=2):
        self.m = ivy.MaxPool2D(k, k, 'Valid', data_format='NCHW') # dataformat: channel comes last in ivy, second in torch
        super(MP, self).__init__()
    def _forward(self, x):
        return self.m(x)

class SP(ivy.Module):
    def __init__(self, k=3, s=1):
        self.m = ivy.MaxPool2D(k, s, k // 2)
        super(SP, self).__init__()

    def _forward(self, x):
        return self.m(x)

class RepConv(ivy.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        self.deploy = deploy
        #self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p)[0][0] == 1


        self.act = ivy.SiLU() if act is True else (act if isinstance(act, ivy.Module) else ivy.Identity())

        if deploy:
            self.rbr_reparam = ivy.Conv2D(c1, c2, [k,k], s, autopad(k, p), with_bias=True, data_format="NCHW")

        else:
            self.rbr_identity = (ivy.BatchNorm2D(c1, data_format="NCHW") if c2 == c1 and s == 1 else None)

            self.rbr_dense = ivy.Sequential(
                ivy.Conv2D(c1, c2, [k,k], s, autopad(k, p), with_bias=False, data_format="NCHW"),
                ivy.BatchNorm2D(c2, data_format="NCHW"),
            )
            #groups missing
            self.rbr_1x1 = ivy.Sequential(
                ivy.Conv2D( c1, c2, [1,1], s, [[0,0]], with_bias=False, data_format="NCHW"),
                ivy.BatchNorm2D(c2, data_format="NCHW"),
            )
        super(RepConv, self).__init__()

    def _forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return ivy.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, ivy.Sequential):
            kernel = branch[0].v['w']
            running_mean = branch[1].v['running_mean']
            running_var = branch[1].v['runnning_var']
            gamma = branch[1].v['w']
            beta = branch[1].v['b']
            eps = branch[1]._eps
        else:
            assert isinstance(branch, ivy.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = ivy.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=ivy.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                #self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
                self.id_tensor = ivy.to_device(kernel_value, branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.v['running_mean']
            running_var = branch.v['running_var']
            gamma = branch.v['w']
            beta = branch.v['b']
            eps = branch._eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


    def fuse_conv_bn(self, conv, bn):

        std = (bn.v['running_var'] + bn.v['eps']).sqrt()
        bias = bn.v['b'] - bn.v['running_mean'] * bn.v['w'] / std

        t = (bn.v['w'] / std).reshape(-1, 1, 1, 1)
        weights = conv.v['w'] * t

        bn = ivy.Identity()
        conv = ivy.Conv2D(in_channels = conv._input_channels,
                              out_channels = conv._output_channels,
                              kernel_size = conv._filter_shape,
                              stride=conv._strides,
                              padding = conv._padding,
                              dilation = conv.dilation,
                              groups = conv.groups,
                              bias = True,
                              padding_mode = conv.padding_mode, data_format="NCHW")

        conv.v['w'] = weights
        conv.v['b'] = bias
        return conv

class ReOrg(ivy.Module):
    def __init__(self):
        super(ReOrg, self).__init__()

    def _forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return ivy.concat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], axis=1)

class Shortcut(ivy.Module):
    def __init__(self, dimension=0):
        self.d = dimension
        super(Shortcut, self).__init__()

    def _forward(self, x):
        return x[0]+x[1]

class Concat(ivy.Module):
    def __init__(self, dimension=1):
        self.d = dimension
        super(Concat, self).__init__()

    def _forward(self, x):
      #for i in x:
        #print("x, d", i.shape, self.d)
      #print()
      return ivy.concat(x, axis=self.d)

class Upsample(ivy.Module):
  def __init__(self, size, scale_factor, mode='linear'):
    self.size = size
    self.scale_factor = scale_factor
    self.mode = mode
    super(Upsample, self).__init__()

  def _forward(self, x):
    #print("upsample ", x.shape, self.size, self.mode)
    return ivy.interpolate(x, self.size, mode=self.mode, scale_factor=self.scale_factor)

class SPPCSPC(ivy.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, [1,1], 1)
        self.cv2 = Conv(c1, c_, [1,1], 1)
        self.cv3 = Conv(c_, c_, [3,3], 1)
        self.cv4 = Conv(c_, c_, [1,1], 1)
        self.m = [ivy.MaxPool2D([x,x], 1, x // 2) for x in k]
        self.cv5 = Conv(4 * c_, c_, [1,1], 1)
        self.cv6 = Conv(c_, c_, [3,3], 1)
        self.cv7 = Conv(2 * c_, c2, [1,1], 1)
        super(SPPCSPC, self).__init__()

    def _forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(ivy.concat([x1] + [m(x1) for m in self.m], axis=1)))
        y2 = self.cv2(x)
        return self.cv7(ivy.concat((y1, y2), axis=1))

class ImplicitA(ivy.Module):
    def __init__(self, channel, mean=0., stddev=.02):
        self._mean = mean
        self._stddev = stddev
        self._channel = channel
        super(ImplicitA, self).__init__()
        #self.implicit = self._create_variables()


    def _create_variables(self, device, dtype=None):
        # TODO: replace this with RandomNormal Initializer once it's added to stateful
        return {"implicit": ivy.random_normal(mean = self._mean, std = self._stddev, shape=(1, self._channel, 1, 1), dtype=dtype, device=device)}


    def _forward(self, x):
        return self.v.implicit + x

class ImplicitM(ivy.Module):
    def __init__(self, channel, mean=0., stddev=.02):
        self._mean = mean
        self._stddev = stddev
        self._channel = channel
        super(ImplicitM, self).__init__()
        #self.implicit = self._create_variables()


    def _create_variables(self, device, dtype=None):
        # TODO: replace this with RandomNormal Initializer once it's added to stateful
        return {"implicit": ivy.random_normal(mean = self._mean, std = self._stddev, shape=(1, self._channel, 1, 1), dtype=dtype, device=device)}


    def _forward(self, x):
        return self.v.implicit * x

class DownC(ivy.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, n=1, k=2):
        c_ = int(c1)  # hidden channels
        self.cv1 = Conv(c1, c_, [1,1], 1)
        self.cv2 = Conv(c_, c2//2, [3,3], k)
        self.cv3 = Conv(c1, c2//2, [1,1], 1)
        self.mp = ivy.MaxPool2D([k,k] ,k, 0)
        super(DownC, self).__init__()

    def _forward(self, x):
        return ivy.concat((self.cv2(self.cv1(x)), self.cv3(self.mp(x))), axis=1)

class Detect(ivy.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=(), training=True, deploy=False):  # detection layer
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [ivy.zeros(1)] * self.nl  # init grid
        a = ivy.astype(ivy.array(anchors).reshape((self.nl, -1, 2)), ivy.float32)
        self.anchors = a
        self.anchor_grid = ivy.reshape(a, (self.nl, 1, -1, 1, 1, 2))
        # self.register_buffer('anchors', a)  # shape(nl,na,2)
        # self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = [ivy.Conv2D(x, self.no * self.na, [1, 1], 1, 'Valid', data_format="NCHW") for x in ch] # output conv
        self.training = training
        self.deploy = deploy
        super(Detect, self).__init__()

    def _forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].reshape((bs, self.na, self.no, ny, nx)).permute_dims((0, 1, 3, 4, 2))

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = ivy.to_device(self._make_grid(nx, ny) ,x[i].device)
                y = x[i].sigmoid()

                # xy, wh, conf = y.split(num_or_size_splits=[2, 2, self.nc + 1], axis=4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                # xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                # wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                # y = ivy.concat((xy, wh, conf), axis=4)
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.reshape((bs, -1, self.no)))

        if self.training:
            out = x
        elif self.end2end:
            out = ivy.concat(z, axis=1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = ivy.concat(z, axis=1)
        else:
            out = (ivy.concat(z, axis=1), x)

        return out

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = ivy.meshgrid(ivy.arange(ny), ivy.arange(nx))
        return ivy.astype(ivy.stack((xv, yv), axis=2).reshape((1, 1, ny, nx, 2)), ivy.float32)

    def convert(self, z):
        z = ivy.concat(z, axis=1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = ivy.array([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=ivy.float32,
                                           device=z.device)
        box @= convert_matrix
        return (box, score)

class IDetect(ivy.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(),  ch=(), training=True, deploy=False):  # detection layer
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na =  3 #len(anchors[0]) // 2  # number of anchors
        self.grid = [ivy.zeros((1))] * self.nl  # init grid
        a = ivy.reshape(ivy.astype(ivy.array(anchors), ivy.float32), (self.nl, -1, 2))
        self.anchors = a
        self.anchor_grid = ivy.reshape(ivy.copy_array(a), (self.nl, 1, -1, 1, 1, 2))
        #self.register_buffer('anchors', a)  # shape(nl,na,2)
        #self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = [ivy.Conv2D(x, self.no * self.na, [1, 1], 1, 'Valid', data_format="NCHW") for x in ch]  # output conv
        self.training = training
        self.deploy=deploy
        self.ia = [ImplicitA(x) for x in ch]
        self.im = [ImplicitM(self.no * self.na) for _ in ch]
        super(IDetect, self).__init__()

    def _forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            #x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            #todo don't have contiguous method in ivy arrays
            x[i] = ivy.permute_dims(ivy.reshape(x[i],(bs, self.na, self.no, ny, nx)), (0, 1, 3, 4, 2))

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = ivy.to_device(self._make_grid(nx,ny), x[i].device)
                    #self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(ivy.reshape(y,(bs, -1, self.no)))

        return x if self.training else (ivy.concat(z, axis=1), x)

    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            #x[i] = x[i].reshape(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x[i] = ivy.permute_dims(ivy.reshape(x[i],(bs, self.na, self.no, ny, nx)), (0, 1, 3, 4, 2))

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = ivy.to_device(self._make_grid(nx, ny), (x[i].device))

                y = x[i].sigmoid()
                xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                y = ivy.concat((xy, wh, conf), axis=4)
                z.append(y.reshape((bs, -1, self.no)))

        if self.training:
            out = x
        elif self.end2end:
            out = ivy.concat(z, axis=1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = ivy.concat(z, axis=1)
        else:
            out = (ivy.concat(z, axis=1), x)

        return out

    def fuse(self):
        print("IDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1,c2,_,_ = self.m[i].v.w.shape
            c1_,c2_, _,_ = self.ia[i].v.implicit.shape
            self.m[i].bias += ivy.matmul(self.m[i].v.w.reshape((c1,c2)),self.ia[i].v.implicit.reshape((c2_,c1_))).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1,c2, _,_ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape((c2))
            self.m[i].weight *= ivy.permute_dims(self.im[i].v.implicit, (1,0, *range(2, len(self.im[i].v.implicit.shape))))

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = ivy.meshgrid([ivy.arange(ny), ivy.arange(nx)])
        return ivy.astype(ivy.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)), ivy.float32)

    def convert(self, z):
        z = ivy.concat(z, axis=1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = ivy.array([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=ivy.float32,
                                           device=z.device)
        box @= convert_matrix
        return (box, score)

class IAuxDetect(ivy.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=(), **kwargs):  # detection layer
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [ivy.zeros(1)] * self.nl  # init grid
        a = ivy.reshape(ivy.astype(anchors, ivy.float32), (self.nl, -1, 2))
        self.anchors = a  # shape(nl,na,2)
        self.anchor_grid = ivy.reshape(ivy.copy_array(a), (self.nl, 1, -1, 1, 1, 2))
        # self.register_buffer('anchors', a)  # shape(nl,na,2)
        # self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = [ivy.Conv2D(i, self.no * self.na, [1,1], 1, [[0,0]]) for i in range(self.nl)]  # output conv
        self.m2 = [ivy.Conv2D(i, self.no * self.na, [1,1], 1, [[0,0]]) for i in range(self.nl)]  # output conv

        self.ia = [ImplicitA(x) for x in ch[:self.nl]]
        self.im = [ImplicitM(self.no * self.na) for _ in ch[:self.nl]]
        super(IAuxDetect, self).__init__(**kwargs)

    def _forward(self, x):
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].reshape(bs, self.na, self.no, ny, nx).permute_dims(0, 1, 3, 4, 2)
            # .contiguous() add contiguous support

            x[i+self.nl] = self.m2[i](x[i+self.nl])
            x[i+self.nl] = x[i+self.nl].reshape((bs, self.na, self.no, ny, nx)).permute_dims(0, 1, 3, 4, 2)
            # .contiguous() add contiguous support

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = ivy.to_device(self._make_grid(nx, ny), x[i].device)

                y = x[i].sigmoid()

                xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                y = ivy.concat((xy, wh, conf), axis=4)
                z.append(y.reshape((bs, -1, self.no)))

        return x if self.training else (ivy.concat(z, axis=1), x[:self.nl])

    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = ivy.permute_dims(x[i].reshape((bs, self.na, self.no, ny, nx)),(0, 1, 3, 4, 2))
            #.contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = ivy.to_device(self._make_grid(nx, ny), x[i].device)

                y = x[i].sigmoid()
                xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].data  # wh
                y = ivy.concat((xy, wh, y[..., 4:]), axis=-1)
                z.append(y.reshape((bs, -1, self.no)))

        if self.training:
            out = x
        elif self.end2end:
            out = ivy.concat(z, axis=1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = ivy.concat(z, axis=1)
        else:
            out = (ivy.concat(z, axis=1), x)

        return out

    def fuse(self):
        print("IAuxDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1,c2,_,_ = self.m[i].v.w.shape
            c1_,c2_, _,_ = self.ia[i].v.implicit.shape
            self.m[i].v.b += ivy.matmul(self.m[i].v.w.reshape((c1,c2)),self.ia[i].v.implicit.reshape((c2_,c1_)).squeeze(1))

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1,c2, _,_ = self.im[i].v.implicit.shape
            self.m[i].v.b *= self.im[i].v.implicit.reshape((c2))
            self.m[i].v.w *= self.im[i].v.implicit.permute_dims((0,1, *(range(2, len(self.im[i].v.implicit.shape)))))

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = ivy.meshgrid([ivy.arange(ny), ivy.arange(nx)])
        return ivy.astype(ivy.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)),ivy.float32)

    def convert(self, z):
        z = ivy.concat(z, axis=1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = ivy.array([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=ivy.float32,
                                           device=z.device)
        box @= convert_matrix
        return (box, score)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = [[k // 2, k//2]] if isinstance(k, int) else [[x // 2 for x in k]]  # auto-pad
    return p

def check_anchor_order(m):
    a = m.anchor_grid.prod(axis=-1).flatten()  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)

def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = ivy.interpolate(img,s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = [ivy.ceil(x * ratio / gs) * gs for x in (h, w)]
        return ivy.pad(img, [0, w - s[1], 0, h - s[0]], constant_values=0.447)  # value = imagenet mean

def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = ivy.Conv2D(conv._input_channels,
                          conv._output_channels,
                          conv._filter_shape,
                          conv._stride,
                          conv._padding,
                          # groups=conv.groups,
                          with_bias=True)
                          # .requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = ivy.reshape(conv.v.w.clone(), (conv._output_channels, -1))
    w_bn = ivy.diag(bn.v.w.divide(ivy.sqrt(bn._epsilon + bn.v.running_var)))
    fusedconv.v.w = ivy.copy_array(ivy.reshape(ivy.matmul(w_bn, w_conv), (fusedconv.v.w.shape)))

    # prepare spatial bias
    b_conv = ivy.zeros(ivy.reshape(conv.v.w, (0)), device=conv.v.w.device) if conv.v.b is None else conv.v.b
    b_bn = bn.v.b - bn.v.w.mul(bn.v.running_mean).divide(ivy.sqrt(bn.v.running_var + bn._epsilon))
    fusedconv.v.b = ivy.copy_array(ivy.matmul(w_bn, ivy.reshape(ivy.reshape(b_conv, (-1, 1)), (-1)) + b_bn))

    return fusedconv

def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def parse_model(d, ch):  # model_dict, input_channels(3)
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2)
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [ivy.Conv2D, Conv, RepConv, DownC,
                 SPPCSPC]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [DownC, SPPCSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is ivy.BatchNorm2D:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m in [Detect, IDetect, IAuxDetect,]:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            c2 = ch[f] * 4
        else:
            c2 = ch[f]

        m_ = ivy.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module ty
        #for x in m_.v.cont_to_iterator():
         # print(x)
        np = sum([len(x) for x in m_.v.cont_to_iterator()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return ivy.Sequential(*layers), sorted(save)
