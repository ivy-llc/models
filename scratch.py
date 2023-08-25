from ivy_models.helpers import load_torch_weights
from ivy_models.regnet.regnet import RegNet
from ivy_models.regnet.layers import BlockParams

# num_classes = 1000
# dropout = 0.5
# data_format = "NCHW"
# v=None
# pretrained=True


num_classes = 1000
stem_width = 32
pretrained = True

depth = 18
w_0 = 200
w_a = 106.23
w_m = 2.48
group_width = 112

_block_params = BlockParams.from_init_params(
    depth=depth,
    w_0=w_0,
    w_a=w_a,
    w_m=w_m,
    group_width=group_width,
)


# model = SqueezeNet(
#     "1_0", num_classes, dropout, data_format=data_format, v=v
# )

model = RegNet(_block_params, num_classes, stem_width)


# url = "https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth"
# url = "https://download.pytorch.org/models/regnet_y_400mf-c65dace8.pth"
# url = "https://download.pytorch.org/models/regnet_y_3_2gf-b5a9779c.pth"
url = "https://download.pytorch.org/models/regnet_y_1_6gf-b11a554e.pth"

# trial = 0


def _torch_weights_mapping(old_key, new_key):
    # global trial
    # print(f"========# {trial} ============\n")
    # print(f"====== OLD KEY ======\n{old_key}\n\n")
    # print(f"====== NEW KEY ======\n{new_key}\n")
    # trial += 1

    new_mapping = new_key
    # if "weight" in old_key:
    #     new_mapping = {"key_chain": new_key, "pattern": "b c h w -> h w c b"}
    # elif "bias" in old_key:
    #     new_mapping = {"key_chain": new_key, "pattern": "h -> 1 h 1 1"}

    return new_mapping


w_clean = load_torch_weights(
    url,
    model,
    raw_keys_to_prune=["num_batches_tracked"],
    custom_mapping=_torch_weights_mapping,
)

print(w_clean)
