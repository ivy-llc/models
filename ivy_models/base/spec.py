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

    def from_pretrained(
        self,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ):
        pass

    def to_dict(self):
        return self.__dict__

    def from_dict(self, config_dict: dict):
        if not isinstance(config_dict, dict):
            raise ivy.exceptions.IvyException("`config_dict` must be a Python dict.")

        self.__init__(**config_dict)

    def to_json_file(self, save_directory: str = "."):
        self.save_pretrained(save_directory=save_directory)

    def from_json_file(self, json_file: str):
        if not isinstance(json_file, str) or json_file[-5:] != ".json":
            raise ivy.exceptions.IvyException(
                "`json_file`: {} must be a string with `.json` extension.".format(json)
            )

        try:
            with open(json_file) as f:
                config_dict = json.loads(f)
        except json.JSONDecodeError:
            raise ivy.exceptions.IvyException("File is not a valid JSON document.")

        self.__init__(**config_dict)
