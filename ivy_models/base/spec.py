import ivy
import os
import json
from typing import Union, Optional


class BaseSpec:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError:
                raise ivy.exceptions.IvyException(
                    f"Can't set {key} with value {value} for {self}"
                )

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
        from transformers import PretrainedConfig

        hf_config = PretrainedConfig(**self.__dict__)
        print("Pushing to Hugging Face:", hf_repo_id)
        hf_config.push_to_hub(
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

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        **kwargs,
    ):
        if os.path.isfile(save_directory):
            raise ivy.exceptions.IvyException(
                "`save directory` must be a directory, not a file!"
            )

        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.__dict__, f)
        print("Saved to directory:", save_directory)

        if push_to_hub:
            self.push_to_hf_hub(**kwargs)

    def from_pretrained(self):
        pass

    def to_dict(self):
        pass

    def from_dict(self):
        pass

    def to_json_file(self):
        pass

    def from_json_file(self):
        pass
