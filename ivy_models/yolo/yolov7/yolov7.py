import urllib.request
import os
import subprocess
import sys
import git
from pathlib import Path
import torch
from copy import deepcopy
import ivy
import ivy_models

from .layers import * 
class Model(ivy.Module):
    def __init__(self, cfg='yolor_cgf.yml', ch=3, nc=None, anchors=None, anchor_grid=None, v=None, device=None):  # model, input channels, number of classes
        self.traced = False
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        self._anchors = anchors
        self._anchor_grid = anchor_grid
        if nc and nc != self.yaml['nc']:
            self.yaml['nc'] = nc  # override yaml value
        if anchors is not None:
            self.yaml['anchors'] = anchors  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model._submodules[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            print("before", m.stride)
            m.stride = ivy.array([s / x.shape[-2] for x in self._forward(ivy.zeros((1, ch, s, s)))])  # forward
            print("after", m.stride)
            check_anchor_order(m)
            m.anchors /= ivy.reshape(m.stride, (-1, 1, 1))
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())
        if isinstance(m, IDetect):
            s = 256  # 2x min stride
            if self._anchors is not None:
              m.anchors = self._anchors
              m.anchor_grid = self._anchor_grid
            m.stride = ivy.array([s / x.shape[-2] for x in self._forward(ivy.zeros((1, ch, s, s)))])  # forward
            check_anchor_order(m)
            m.anchors /= ivy.reshape(m.stride, (-1, 1, 1))
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())
        if isinstance(m, IAuxDetect):
            s = 256  # 2x min stride
            m.stride = ivy.array([s / x.shape[-2] for x in self._forward(ivy.zeros((1, ch, s, s)))[:4]])  # forward
            #print(m.stride)
            check_anchor_order(m)
            m.anchors /= ivy.reshape(m.stride, (-1, 1, 1))
            self.stride = m.stride
            self._initialize_aux_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        super(Model, self).__init__(v=v, device=device)

    def _forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self._forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return ivy.concat(y, axis=1), None  # augmented inference, train
        else:
            return self._forward_once(x, profile)  # single-scale inference, train

    def _forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model._submodules:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if not hasattr(self, 'traced'):
                self.traced=False

            if self.traced:
                if isinstance(m, Detect) or isinstance(m, IDetect) or isinstance(m, IAuxDetect):
                    break
            #print('model ', m)
            #if hasattr(x, 'shape'):
              #print(' x shape', x.shape)

            # print('mv', m.v)
            # print("x, m", x, m)
            x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model._submodules[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = ivy.reshape(mi.v.b, (m.na, -1)) # conv.bias(255) to (3,85)
            #print(b)
            b[:, 4] += ivy.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += ivy.log(0.6 / (m.nc - 0.99)) if cf is None else ivy.log(cf / cf.sum())  # cls
            b = b.flatten()
            mi.v.b = mi.v.b.reshape((1, len(b), 1, 1))

    def _initialize_aux_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model._submodules[-1]  # Detect() module
        for mi, mi2, s in zip(m.m, m.m2, m.stride):  # from
            b = ivy.reshape(mi.v.b, (m.na, -1))  # conv.bias(255) to (3,85)
            b.data[:, 4] += ivy.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += ivy.log(0.6 / (m.nc - 0.99)) if cf is None else ivy.log(cf / cf.sum())  # cls
            mi.v.b = ivy.reshape(b, (-1))
            b2 = ivy.reshape(mi2.v.b, (m.na, -1))  # conv.bias(255) to (3,85)
            b2.data[:, 4] += ivy.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b2.data[:, 5:] += ivy.log(0.6 / (m.nc - 0.99)) if cf is None else ivy.log(cf / cf.sum())  # cls
            mi2.v.b = b2.flatten()

    def _initialize_biases_bin(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model._submodules[-1]  # Bin() module
        bc = m.bin_count
        for mi, s in zip(m.m, m.stride):  # from
            b = ivy.reshape(mi.v.b, (m.na, -1))  # conv.bias(255) to (3,85)
            old = b[:, (0,1,2,bc+3)].data
            obj_idx = 2*bc+4
            b[:, :obj_idx].data += ivy.log(0.6 / (bc + 1 - 0.99))
            b[:, obj_idx].data += ivy.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, (obj_idx+1):].data += ivy.log(0.6 / (m.nc - 0.99)) if cf is None else ivy.log(cf / cf.sum())  # cls
            b[:, (0,1,2,bc+3)].data = old
            mi.v.b = b.flatten()

    def _initialize_biases_kpt(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model._submodules[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.v.b.reshape((m.na, -1))  # conv.bias(255) to (3,85)
            b.data[:, 4] += ivy.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += ivy.log(0.6 / (m.nc - 0.99)) if cf is None else ivy.log(cf / cf.sum())  # cls
            mi.v.b = b.flatten()

    def _print_biases(self):
        m = self.model._submodules[-1]  # Detect() module
        for mi in m.m:  # from
            #skipped mi.v.b.detach()
            b = mi.v.b.reshape((m.na, -1)).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.v.w.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))


    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, RepConv):
                #print(f" fuse_repvgg_block")
                m.fuse_repvgg_block()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
            elif isinstance(m, (IDetect, IAuxDetect)):
                m.fuse()
                m.forward = m.fuseforward
        self.info()
        return self

def YOLOv7_torch_ivy_mapping(old_key, new_key):
    new_mapping = new_key
    if "conv/weight" in old_key:
      new_mapping = {"key_chain": new_key, "pattern": "b c h w-> h w c b"}
    elif "rbr_1x1" in old_key:
      if "0/weight" in old_key:
        new_mapping = {"key_chain": new_key, "pattern": "b c h w-> h w c b" }
    elif "rbr_dense" in old_key:
      if "0/weight" in old_key:
        new_mapping = {"key_chain": new_key, "pattern": "b c h w-> h w c b" }
    elif "m/0/weight" in old_key or "m/1/weight" in old_key or "m/2/weight" in old_key:
       new_mapping = {"key_chain": new_key, "pattern": "b c h w-> h w c b" }
    elif "m/0/bias" in old_key or "m/1/bias" in old_key or "m/2/bias" in old_key:
       new_mapping = {"key_chain": new_key, "pattern": "c-> 1 c 1 1" }

    return new_mapping


def yolov7(cfg = './ivy_models/yolo/yolov7/config/yolov7.yaml', pretrained=True):
    if not pretrained:
        return Model(cfg=cfg)
    urllib.request.urlretrieve('https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt', './yolov7_training.pt')
    repo_url = 'https://github.com/WongKinYiu/yolov7'
    repo_dir = './yolov7'
    git.Repo.clone_from(repo_url, repo_dir)
    # Change directory to 'yolov7'
    original_directory = os.getcwd()
    os.chdir('yolov7')

    # Install the required packages
    subprocess.call(['pip', 'install', '-r', 'requirements.txt'])
    # Go back to the original directory
    os.chdir(original_directory)
    torch_model = torch.hub.load('./yolov7', 'custom', './yolov7_training.pt',force_reload=True, source='local',trust_repo=True)
    weights = torch.hub.load_state_dict_from_url('https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt')

    torch_dict = weights['model'].state_dict().copy()
    anchor_grid = weights['model'].state_dict()['model.105.anchor_grid']
    anchors = weights['model'].state_dict()['model.105.anchors']
    torch_dict = {key: value for key, value in torch_dict.items() if not (key.endswith("num_batches_tracked")or key.endswith("anchors") or key.endswith("anchor_grid"))}
    reference_model = Model(cfg=cfg)
    w_clean = ivy_models.helpers.load_torch_weights(
        weights=torch_dict, ref_model=reference_model.model, custom_mapping=YOLOv7_torch_ivy_mapping
    )

    # Delete the repository directory
    os.system('rm -rf ./yolov7')

    # Delete the yolov7_training.pt file
    os.remove('./yolov7_training.pt')

    return Model(cfg=cfg, anchors=ivy.asarray(anchors), anchor_grid=ivy.asarray(anchor_grid), v = ivy.Container({'model':w_clean}))