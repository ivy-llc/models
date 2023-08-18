import ivy
import os
import inspect
from typing import Optional


class abstractclassmethod(classmethod):
    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(callable)


class BaseModel(ivy.Module):
    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__(*args, **kwargs)

    def __setattr__(self, key, value):
        prev_call = inspect.getframeinfo(inspect.currentframe().f_back)[0]
        from_test = "ivy_models_tests" in prev_call
        from_ivy_module = "ivy/stateful/module" in prev_call
        if (
            key == "v"
            and "v" in self.__dict__.keys()
            and self.__dict__["v"] is not None
            and not (from_test or from_ivy_module)
        ):
            ivy.Container.cont_assert_identical_structure([self.v, value])
        self.__dict__[key] = value

    @abstractclassmethod
    def get_spec_class(self):
        raise NotImplementedError()

    def _hf_verify_or_login(self):
        from huggingface_hub import login, HfFolder

        folder = HfFolder()
        if folder.get_token() is None:
            login()

    def push_to_huggingface(
        self,
        repo_id: str,
        config_path: str = "config.json",
        model_path: str = "model.pkl",
        weights_path: str = "weights.hdf5",
        repo_type: str = "model",
        token: Optional[str] = None,
        private: bool = False,
        revision: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        create_pr: bool = False,
        safe_serialization: bool = False,
        push_config: bool = True,
        push_model: bool = True,
        push_weights: bool = True,
    ):
        from huggingface_hub import HfApi

        self._hf_verify_or_login()

        api = HfApi()
        api.create_repo(
            repo_id, token=token, private=private, repo_type=repo_type, exist_ok=True
        )

        if push_config:
            print("Pushing config to Hugging Face...")
            self.spec.to_json_file()
            api.upload_file(
                path_or_fileobj=config_path,
                repo_id=repo_id,
                path_in_repo=config_path,
                repo_type=repo_type,
            )
            os.remove(config_path)

        if push_model:
            print("Pushing model object to Hugging Face...")
            self.save(model_path)
            api.upload_file(
                path_or_fileobj=model_path,
                repo_id=repo_id,
                path_in_repo=model_path,
                repo_type=repo_type,
            )
            os.remove(model_path)

        if push_weights:
            print("Pushing model weights to Hugging Face...")
            self.v.cont_to_disk_as_hdf5(weights_path)
            api.upload_file(
                path_or_fileobj=weights_path,
                repo_id=repo_id,
                path_in_repo=weights_path,
                repo_type=repo_type,
            )
            os.remove(weights_path)

        print("Successful!")

    def save_pretrained(
        self,
        config_path: str = "config.json",
        model_path: str = "model.pkl",
        weights_path: str = "weights.hdf5",
        save_config: bool = True,
        save_model: bool = True,
        save_weights: bool = True,
    ):
        if save_config:
            print("Saving config...")
            self.spec.to_json_file()

        if save_model:
            print("Saving model object...")
            self.save(model_path)

        if save_weights:
            print("Saving model weights...")
            self.v.cont_to_disk_as_hdf5(weights_path)

        print("Successful!")

    @classmethod
    def load_from_huggingface(
        self,
        repo_id: str,
        config_path: str = "config.json",
        model_path: str = "model.pkl",
        weights_path: str = "weights.hdf5",
        repo_type: str = "model",
        token: Optional[str] = None,
        revision: Optional[str] = None,
        safe_serialization: bool = False,
        load_model_object: bool = False,
    ):
        from huggingface_hub import hf_hub_download

        if load_model_object:
            hf_hub_download(
                filename=model_path,
                repo_id=repo_id,
                repo_type="model",
                local_dir=".",
            )
            obj = self.load(model_path)
            os.remove(model_path)
            return obj

        else:
            hf_hub_download(
                filename=weights_path,
                repo_id=repo_id,
                repo_type="model",
                local_dir=".",
            )
            weights = ivy.Container.cont_from_disk_as_hdf5(weights_path)
            os.remove(weights_path)

            hf_hub_download(
                filename=config_path,
                repo_id=repo_id,
                repo_type="model",
                local_dir=".",
            )
            spec = self.get_spec_class().from_json_file(config_path)
            os.remove(config_path)

            model = self(spec=spec)
            model.v = weights

            return model
