import ivy
import pytest
import numpy as np
from ivy_models.roberta import roberta_base


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_roberta(device, fw, batch_shape, load_weights):
    """Test RoBerta Base Sequence Classification"""

    num_dims = 768
    inputs = np.load(
        "/models/ivy_models_tests/roberta/roberta_inputs.npy", allow_pickle=True
    ).tolist()

    model = roberta_base(load_weights)
    inputs = {k: ivy.asarray(v) for k, v in inputs.items()}
    logits = model(**inputs)["pooler_output"]
    ref_logits = np.load("/models/ivy_models_tests/roberta/roberta_pooled_output.npy")
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_dims])
    assert np.allclose(ref_logits, logits, rtol=0.005, atol=0.0005)
