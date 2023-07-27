import ivy
import os
import json
from typing import Optional


class BaseSpec:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError:
                raise ivy.exceptions.IvyException(
                    f"Can't set {key} with value {value} for {self}"
                )

    def _hf_verify_or_login(self):
        from huggingface_hub import login, HfFolder

        folder = HfFolder()
        if folder.get_token() is None:
            login()

    def push_to_huggingface(
        self,
        repo_id: str,
        config_path: str = "config.json",
        repo_type: str = "model",
        token: Optional[str] = None,
        private: bool = False,
        revision: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        create_pr: bool = False,
        safe_serialization: bool = False,
    ):
        from huggingface_hub import HfApi

        self._hf_verify_or_login()

        api = HfApi()
        api.create_repo(
            repo_id, token=token, private=private, repo_type=repo_type, exist_ok=True
        )

        print("Pushing config to Hugging Face...")
        self.to_json_file()
        api.upload_file(
            path_or_fileobj=config_path,
            repo_id=repo_id,
            path_in_repo=config_path,
            repo_type=repo_type,
        )
        os.remove(config_path)
        print("Successful!")

    def save_pretrained(
        self,
        config_path: str = "config.json",
    ):
        print("Saving config...")
        self.to_json_file()

        print("Successful!")

    def to_dict(self):
        return self.__dict__

    def to_json_file(self, save_directory: str = "."):
        if os.path.isfile(save_directory):
            raise ivy.exceptions.IvyException(
                "`save directory` must be a directory, not a file!"
            )

        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.__dict__, f)
        print("Saved to directory:", save_directory)

    @classmethod
    def load_from_huggingface(
        self,
        repo_id: str,
        config_path: str = "config.json",
        repo_type: str = "model",
        token: Optional[str] = None,
        revision: Optional[str] = None,
        safe_serialization: bool = False,
    ):
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            filename=config_path,
            repo_id=repo_id,
            repo_type="model",
            local_dir=".",
        )
        spec = self.from_json_file(config_path)
        os.remove(config_path)
        return spec

    @classmethod
    def from_dict(self, config_dict: dict):
        if not isinstance(config_dict, dict):
            raise ivy.exceptions.IvyException("`config_dict` must be a Python dict.")
        return self(**config_dict)

    @classmethod
    def from_json_file(self, json_file: str):
        if not isinstance(json_file, str) or json_file[-5:] != ".json":
            raise ivy.exceptions.IvyException(
                "`json_file`: {} must be a string with `.json` extension.".format(json)
            )

        try:
            with open(json_file) as f:
                config_dict = json.loads(f.read())
        except json.JSONDecodeError:
            raise ivy.exceptions.IvyException("File is not a valid JSON document.")

        return self(**config_dict)
