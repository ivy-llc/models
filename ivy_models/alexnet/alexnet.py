import ivy


class AlexNet(ivy.Module):
    """An Ivy native implementation of AlexNet"""

    def __init__(self, num_classes=1000, dropout=0, v=None):
        self.num_classes = num_classes
        self.dropout = dropout
        super(AlexNet, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.features = ivy.Sequential(
            ivy.Conv2D(3, 64, [11, 11], [4, 4], 2, data_format="NCHW"),
            ivy.ReLU(),
            ivy.MaxPool2D(3, 2, 0, data_format="NCHW"),
            ivy.Conv2D(64, 192, [5, 5], [1, 1], 2, data_format="NCHW"),
            ivy.ReLU(),
            ivy.MaxPool2D(3, 2, 0, data_format="NCHW"),
            ivy.Conv2D(192, 384, [3, 3], 1, 1, data_format="NCHW"),
            ivy.ReLU(),
            ivy.Conv2D(384, 256, [3, 3], 1, 1, data_format="NCHW"),
            ivy.ReLU(),
            ivy.Conv2D(256, 256, [3, 3], 1, 1, data_format="NCHW"),
            ivy.ReLU(),
            ivy.MaxPool2D(3, 2, 0, data_format="NCHW"),
        )
        self.avgpool = ivy.AdaptiveAvgPool2d((6, 6))
        self.classifier = ivy.Sequential(
            ivy.Dropout(prob=self.dropout),
            ivy.Linear(256 * 6 * 6, 4096),
            ivy.ReLU(),
            ivy.Dropout(prob=self.dropout),
            ivy.Linear(4096, 4096),
            ivy.ReLU(),
            ivy.Linear(4096, self.num_classes),
        )

    def _forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = ivy.reshape(x, (x.shape[0], -1))
        x = self.classifier(x)
        return x


def alexnet(pretrained=True, num_classes=1000, dropout=0):
    """Loads an instance of AlexNet, optionally with pretrained weights."""
    if not pretrained:
        return AlexNet(num_classes=num_classes, dropout=dropout)

    import torch

    weights = torch.hub.load_state_dict_from_url(
        "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
    )
    weights_raw = ivy.Container(weights)

    reference_model = AlexNet(num_classes=1000)
    mapping = {}
    for old_key, new_key in zip(
        weights_raw.cont_sort_by_key().cont_to_iterator_keys(),
        reference_model.v.cont_sort_by_key().cont_to_iterator_keys(),
    ):
        new_mapping = new_key
        if "features" in old_key and "bias" in old_key:
            new_mapping = {"key_chain": new_key, "pattern": "h -> 1 h 1 1"}
        elif "features" in old_key and "weight" in old_key:
            new_mapping = {"key_chain": new_key, "pattern": "b c h w-> h w c b"}
        mapping[old_key] = new_mapping

    w_clean = weights_raw.cont_restructure(mapping, keep_orig=False)
    w_clean = ivy.asarray(w_clean)

    return AlexNet(num_classes=1000, dropout=0, v=w_clean)
