from transformers import AutoTokenizer, AutoModel
import ivy
import pytest
import numpy as np
from ivy_models.bert import bert_base_uncased


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_bert(device, f, fw, batch_shape, load_weights):
    """Test Bert Base Sequence Classification"""

    num_dims = 768
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    inputs = tokenizer("hello, my cat is cute.", return_tensor="pt")
    inputs = (
        {k: v.to(device) for k, v in inputs.items()} if device is not None else inputs
    )
    ref_model = AutoModel.from_pretrained("bert-base-uncased")
    model = bert_base_uncased(load_weights)

    ref_logits = ref_model(**inputs)[0].numpy()
    inputs = {k: ivy.asarray(v.numpy()) for k, v in inputs.items()}
    logits = model(**inputs)["pooled_output"]

    assert logits.shape == tuple([ivy.to_scaler(batch_shape), num_dims])

    assert np.allclose(ref_logits, logits, rtol=0.005, atol=0.0005)

