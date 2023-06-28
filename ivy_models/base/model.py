import ivy
from typing import Union, Optional


class BaseModel(ivy.Module):
    def push_to_hf_hub(
        self,
        hf_repo_id: str,
        use_temp_dir: Optional[bool] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        max_shard_size: Optional[Union[int, str]] = "10GB",
        create_pr: bool = False,
        safe_serialization: bool = False,
    ):
        from transformers import PretrainedConfig, PreTrainedModel

        hf_config = PretrainedConfig(**self._spec.__dict__, torch_dtype="float32")
        print(hf_config.torch_dtype, "\n")

        class IvyHfModel(PreTrainedModel):
            def __init__(cls):
                super().__init__(hf_config)
                cls.model = self
                print(cls.model.config.torch_dtype)

            def forward(cls, *args, **kwargs):
                return cls.model(*args, **kwargs)

        hf_model = IvyHfModel()
        print("Pushing to Hugging Face:", hf_repo_id)
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
