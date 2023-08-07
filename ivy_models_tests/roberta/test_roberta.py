import os
import ivy
import pytest
import numpy as np
from ivy_models import roberta_base


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_roberta(device, fw, batch_shape, load_weights):
    """Test RoBerta Base Sequence Classification"""

    num_dims = 768
    this_dir = os.path.dirname(os.path.realpath(__file__))
    input_path = os.path.join(this_dir, "roberta_inputs.npy")
    inputs = np.load(input_path, allow_pickle=True).tolist()
    model = roberta_base(load_weights)

    inputs = {k: ivy.asarray(v) for k, v in inputs.items()}
    logits = model(**inputs)["pooler_output"]
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_dims])

    if load_weights:
        ref_logits_path = os.path.join(this_dir, "roberta_pooled_output.npy")
        ref_logits = np.load(ref_logits_path)
        assert np.allclose(ref_logits, ivy.to_numpy(logits), rtol=0.005, atol=0.005)
