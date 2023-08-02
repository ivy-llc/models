import ivy
import pytest
import numpy as np
from ivy_models.bert import bert_base_uncased


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_bert(device, f, fw, batch_shape, load_weights):
    """Test Bert Base Sequence Classification"""

    num_dims = 768
    inputs = np.load("bert_inputs.npy").tolist()
    model = bert_base_uncased(load_weights)

    inputs = {k: ivy.asarray(v) for k, v in inputs.items()}
    logits = model(**inputs)["pooler_output"]
    ref_logits = np.load("bert_pooled_output.npy")
    assert logits.shape == tuple([ivy.to_scaler(batch_shape), num_dims])
    assert np.allclose(ref_logits, logits, rtol=0.005, atol=0.0005)
