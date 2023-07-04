import ivy
import ivy_models
import os
from typing import Union, Optional


class BaseModel(ivy.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _get_hf_model(
        self,
        backend: str = "torch",
    ):
        from transformers import (
            PretrainedConfig,
            PreTrainedModel,
            TFPreTrainedModel,
            FlaxPreTrainedModel,
        )

        hf_config = PretrainedConfig(**self._spec.__dict__)

        if backend in ["jax", "flax"]:

            class IvyHfModel(FlaxPreTrainedModel):
                def __init__(cls):
                    ivy.set_backend("jax")
                    # todo: check input shape with config var
                    model = ivy.transpile(
                        self,
                        to="flax",
                        # todo: fix the arg!
                        args=(ivy.random_uniform(shape=(1, 3, 224, 224)),),
                    )
                    super().__init__(hf_config, model)

                def forward(cls, *args, **kwargs):
                    return cls.model(*args, **kwargs)

        elif backend in ["torch", "tensorflow", "keras"]:
            if backend == "keras":
                backend = "tensorflow"
            HF_MODELS = {
                "torch": PreTrainedModel,
                "tensorflow": TFPreTrainedModel,
            }

            class IvyHfModel(HF_MODELS[backend]):
                def __init__(cls):
                    super().__init__(hf_config)
                    ivy.set_backend(backend)
                    # todo: check input shape with config var
                    cls.model = ivy.transpile(
                        self,
                        to=backend,
                        args=(
                            ivy.random_uniform(shape=(1, 224, 224, hf_config.input_dim))
                        ),
                    )

                def forward(cls, *args, **kwargs):
                    return cls.model(*args, **kwargs)

        else:
            raise ivy.exceptions.IvyException(
                "backend: {} must be in [torch, tensorflow, keras, jax, flax]".format(
                    backend
                )
            )

        return IvyHfModel()

    def push_to_hf_hub(
        self,
        hf_repo_id: str,
        backend: str = "torch",
        use_temp_dir: Optional[bool] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        max_shard_size: Optional[Union[int, str]] = "10GB",
        create_pr: bool = False,
        safe_serialization: bool = False,
    ):
        hf_model = self._get_hf_model(backend)

        print("Pushing {} model to Hugging Face: {}", backend, hf_repo_id)
        hf_model.push_to_hub(
            repo_id=hf_repo_id,
            use_temp_dir=use_temp_dir,
            commit_message=commit_message,
            private=private,
            use_auth_token=use_auth_token,
            max_shard_size=max_shard_size,
            create_pr=create_pr,
            safe_serialization=safe_serialization,
        )
        print("Successful!")

    def save_pretrained(self):
        return

    @classmethod
    def from_pretrained(
        self,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[ivy_models.BaseSpec, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ):
        return
