from typing import List, Union

from .decorator import emit
from .messenger import Emitter
from .configs import Config


class Actor:
    """Abstract class for an actor of federated learning.
    An actor can be a node, the central server or the aggregator.
    """

    def __init__(self, **configs: Union[Config, List[Config]]):
        self.emitter_path = None
        self.iterations = 0

        for config in configs:
            if isinstance(configs[config], list):
                for config_ in configs[config]:
                    if not isinstance(config_, Config):
                        raise TypeError(f"One configuration in configuration list '{config}' is of type "
                                        f"{type(config_)} instead of type Config")
            elif not isinstance(configs[config], Config):
                raise TypeError(f"Configuration '{config}' is of type {type(configs[config])}"
                                " instead of type Config")
            else:
                if self.emitter_path is None and "emitter_path" in configs[config].configs:
                    self.emitter_path = configs[config].emitter_path
            setattr(self, config, configs[config])

        if self.emitter_path is None:
            raise ValueError(f"Did not find 'emitter_path' configuration for object of type {self.__class__.__name__}")

        self.emitter = Emitter(self.emitter_path)
        self.create()

    @emit
    def create(self):
        """Implement in daughter class"""
        pass

    @emit
    def run(self, timeout: Union[int, float] = 0, sleeptime: Union[int, float] = 5):
        """Implement in daughter class.
        Monitors changes in the inputs of the actor and triggers it upon changes.

        Parameters
        ----------
        timeout: Union[int, float]
            How many seconds should the run last. If <= 0, will last until killed by the user. Default value = 0.
        sleeptime: Union[int, float]
            How many seconds between each checks. Default value = 5.
        """
        pass
