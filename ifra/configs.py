from copy import copy
from pathlib import Path
from typing import Union

from transparentpath import TransparentPath


class Config:

    """Class to load a node configuration on its local computer. Basically just a wrapper around a json file."""

    EXPECTED_CONFIGS = []

    # noinspection PyUnresolvedReferences
    def __init__(self, path: Union[str, Path, TransparentPath]):

        if type(path) == str:
            path = TransparentPath(path)
        self.path = path

        if not path.suffix == ".json":
            raise ValueError("Config path must be a json")

        if hasattr(path, "read"):
            self.configs = path.read()
        else:
            with open(path) as opath:
                self.configs = load(opath)

        for key in self.EXPECTED_CONFIGS:
            if key not in self.configs:
                raise IndexError(f"Missing required config {key}")
        for key in self.configs:
            if key not in self.EXPECTED_CONFIGS:
                raise IndexError(f"Unexpected config {key}")

        for key in self.configs:
            if self.configs[key] == "":
                self.configs[key] = None

    def __getattr__(self, item):
        if item not in self.configs:
            raise ValueError(f"No configuration named '{item}' was found")
        return self.configs[item]

    def save(self):
        to_save = copy(self.configs)

        for key in to_save:
            if to_save[key] is None:
                to_save[key] = ""

        self.path.write(to_save)


class NodeLearningConfig(Config):
    EXPECTED_CONFIGS = [
        "features_names",
        "classes_names",
        "x_mins",
        "x_maxs",
        "max_depth",
        "remember_activation",
        "stack_activation",
        "plot_data",
        "get_leaf",
        "local_model_path",
        "central_model_path",
        "dataprep_method",
        "id"
    ]

    def __eq__(self, other):
        if not isinstance(other, NodeLearningConfig):
            return False
        for key in NodeLearningConfig.EXPECTED_CONFIGS:
            if key == "local_model_path":  # This parameter can change without problem
                continue
            if self[key] != other[key]:
                return False
        return True


class CentralLearningConfig(Config):
    EXPECTED_CONFIGS = [
        "max_coverage",
        "output_path",
    ]


class Paths(Config):
    EXPECTED_CONFIGS = ["x", "y", "x_read_kwargs", "y_read_kwargs"]

    def __init__(self, path: Union[str, Path, TransparentPath]):
        super().__init__(path)
        for item in self.configs:
            if isinstance(self.configs[item], str):
                self.configs[item] = TransparentPath(self.configs[item])
