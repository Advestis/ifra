from typing import List, Union

from .decorator import emit
from .messenger import Emitter
from .configs import Config


class Actor:
    """Abstract class for an actor of federated learning.
    An actor can be a node, the central server or the aggregator
    """

    def __init__(self, **configs: Union[Config, List[Config]]):
        self.path_emitter = None

        for config in configs:
            if isinstance(configs[config], list):
                for config_ in configs[config]:
                    if not isinstance(config_, Config):
                        raise TypeError(f"One configuration in configuration list '{config}' is of type "
                                        f"{type(config_)} instead of type Config")
            elif not isinstance(configs[config], Config):
                raise TypeError(f"Configuration '{config}' is of type {type(configs[config])}"
                                " instead of type Config")
            setattr(self, config, configs[config])
            if hasattr(configs[config], "path_emitter"):
                self.path_emitter = configs[config].emitter_path

        if self.path_emitter is None:
            raise ValueError(f"Did not find 'path_emitter' configuration for object of type {self.__class__.__name__}")

        self.emitter = Emitter(self.path_emitter)
        self.create()

    @emit
    def create(self):
        """Implement in daughter class"""
        pass

    @emit
    def run(self, timeout: int, sleeptime: int):
        """Implement in daughter class"""
        pass
