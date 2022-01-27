from copy import deepcopy, copy
from pathlib import Path
from time import time, sleep
from typing import Union
from json import load
from transparentpath import TransparentPath
import logging

logger = logging.getLogger(__name__)


class Emitter:
    """Instance to send messages to a receiver. It uses a json file to write messages.

     Uses the same logic than `ifra.configs.Config` to handle the json file that is read or written by
     `ifra.node.Node` and `ifra.central_server.NodeGate`.

     Only messages which names are in `ifra.messenger.Emitter.DEFAULT_MESSAGES` are allowed.

     Attributes
     ----------
     path: TransparentPath
        The path to the json file containing the messages. Will be casted into a transparentpath if a str is
        given, so make sure that if you pass a string that should be local, you did not set a global file system.
        If it does not exist, it is created with the default messages present in
        `ifra.messenger.Emitter.DEFAULT_MESSAGES`. If it exists, it is overwritten and a warning is logged.
     messages: dict
        A dictionnary of messages
     """
    DEFAULT_MESSAGES = {}
    """Overload in daughter classes."""

    def __init__(self, path: Union[str, Path, TransparentPath]):
        """
        Parameters
        ----------
        path: Union[str, Path, TransparentPath]
            The path to the json file containing the messages. Will be casted into a transparentpath if a str is
            given, so make sure that if you pass a str that should be local, you did not set a global file system.
        """

        if type(path) == str:
            path = TransparentPath(path)

        if not path.suffix == ".json":
            raise ValueError("Emitter path must be a json")

        if path.is_file():
            raise ValueError(f"Message file '{path}' existed at Emitter creation. "
                             f"This can indicate a crash of a previous execution, or the fact that two"
                             f"sessions are running at the same time. Delete this file and restart.")

        lockfile = path.append(".locked")
        if lockfile.is_file():
            raise ValueError(f"Lock file '{path}' existed at Emitter creation. "
                             f"This can indicate a crash of a previous execution, or the fact that two"
                             f"sessions are running at the same time. Delete this file and restart.")
        else:
            lockfile.touch()

        self.path = path
        self.messages = {}
        try:
            lockfile.rm(absent="ignore")
            self.reset_messages()
        except Exception as e:
            lockfile.rm(absent="ignore")
            raise e

    def __getattr__(self, item):
        if item == "messages":
            raise ValueError("NodeMessenger object's 'messages' not set")
        if item == "path":
            raise ValueError("NodeMessenger object's 'path' not set")
        if item not in self.messages:
            raise ValueError(f"No messages named '{item}' was found")
        return self.messages[item]

    def __setattr__(self, item, value):
        if item == "messages" or item == "path":
            super().__setattr__(item, value)
            return
        if item not in self.messages and item not in self.__class__.DEFAULT_MESSAGES:
            raise ValueError(f"Message named '{item}' is not allowed")
        self.messages[item] = value
        self.save()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        setattr(result, "messages", deepcopy(self.messages, memo))
        return result

    def save(self) -> None:
        """Saves the current messages into the file it used to load."""
        to_save = copy(self.messages)

        lockfile = self.path.append(".locked")
        t = time()
        while lockfile.is_file():
            sleep(0.5)
            if time() - t > 5:
                raise FileExistsError(f"File {lockfile} exists : could not lock on file {self.path}")
        lockfile.touch()

        try:
            for key in to_save:
                if key not in self.__class__.DEFAULT_MESSAGES:
                    raise ValueError(f"Forbidden message '{key}'")
                if to_save[key] is None:
                    to_save[key] = ""

            self.path.write(to_save)
            lockfile.rm(absent="ignore")
        except Exception as e:
            lockfile.rm(absent="ignore")
            raise e

        lockfile = self.path.append(".locked")
        t = time()
        while lockfile.is_file():
            sleep(0.5)
            if time() - t > 5:
                raise FileExistsError(f"File {lockfile} exists : could not lock on file {self.path}")
        lockfile.touch()

        try:

            if not self.path.is_file():
                raise FileNotFoundError(f"Messenger file {self.path} not found")

            if hasattr(self.path, "read"):
                self.messages = self.path.read()
            else:
                with open(self.path) as opath:
                    self.messages = load(opath)

            for key in self.messages:
                if self.messages[key] == "":
                    self.messages[key] = None

            lockfile.rm(absent="ignore")
        except Exception as e:
            lockfile.rm(absent="ignore")
            raise e

    def reset_messages(self):
        """Reset messages to their default values specified in `ifra.messenger.Emitter.DEFAULT_MESSAGES`"""
        self.messages = copy(self.__class__.DEFAULT_MESSAGES)
        self.save()

    def rm(self):
        """Removes the Emitter's json file."""
        lockfile = self.path.append(".locked")
        t = time()
        while lockfile.is_file():
            sleep(0.5)
            if time() - t > 5:
                raise FileExistsError(f"File {lockfile} exists : could not lock on file {self.path}")
        self.path.rm(absent="ignore")
        lockfile.rm()


class Receiver:
    """Instance to read messages from an emitter. It uses a json file to read messages.

     Uses the same logic than `ifra.configs.Config` to handle the json file.

     Only messages which names are in `ifra.messenger.Receiver.DEFAULT_MESSAGES` are allowed.

     Attributes
     ----------
     path: TransparentPath
        The path to the json file containing the messages. Will be casted into a transparentpath if a str is
        given, so make sure that if you pass a string that should be local, you did not set a global file system.
        If it does not exist, Receiver waits until the file exists, or raises TimeoutError if it waited more than
        `ifra.messenger.Receiver.timeout_when_missing` seconds (default is 600s).
     messages: dict
        A dictionnary of messages
     """
    DEFAULT_MESSAGES = {}
    """Overload in daughter classes. Must match the `ifra.messenger.Emitter.DEFAULT_MESSAGES` of the emitter that
    will write to the file the receiver will read."""

    timeout_when_missing = 600
    """How long the messenger is supposed to wait for its file to come back when using
    `ifra.messenger.NodeMessenger.get_latest_messages`"""

    def __init__(self, path: Union[str, Path, TransparentPath]):
        """
        Parameters
        ----------
        path: Union[str, Path, TransparentPath]
            The path to the json file containing the messages. Will be casted into a transparentpath if a str is
            given, so make sure that if you pass a str that should be local, you did not set a global file system.
        """

        if type(path) == str:
            path = TransparentPath(path)

        if not path.suffix == ".json":
            raise ValueError("Messenger path must be a json")
        self.path = path
        self.messages = {}
        self.new = True
        self.get_latest_messages()

    def __getattr__(self, item):
        if item == "messages":
            raise ValueError("NodeMessenger object's 'messages' not set")
        if item == "path":
            raise ValueError("NodeMessenger object's 'path' not set")
        if item not in self.messages:
            raise ValueError(f"No messages named '{item}' was found")
        return self.messages[item]

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        setattr(result, "messages", deepcopy(self.messages, memo))
        return result

    def get_latest_messages(self):
        """Gets latest messages by reading the messages json file. If file does not exist and it is the first time we
        read it (object init), wait for it for at most `ifra.messenger.Receiver.timeout_when_missing` seconds then
        raises TimeoutError. Else, raises FileNotFoundError."""
        first_message = True

        if self.__class__.timeout_when_missing is None:
            raise ValueError("'timeout_when_missing' can not be None")

        t = time()
        while not self.path.is_file() and time() - t < self.timeout_when_missing:
            if not self.new:
                raise FileNotFoundError(f"Message file {self.path} disapread.")
            if first_message:
                first_message = False
                logger.info(f"Message file {self.path} not here yet. Waiting for it to arrive...")
            sleep(1)
        if not self.path.is_file():
            raise TimeoutError(f"Could not find message file '{self.path}' in less that "
                               f"{self.__class__.timeout_when_missing} seconds. Aborting.")
        if first_message is False:
            logger.info("...message file found. Resuming.")

        self.new = False

        lockfile = self.path.append(".locked")
        t = time()
        while lockfile.is_file():
            sleep(0.5)
            if time() - t > 5:
                raise FileExistsError(f"File {lockfile} exists : could not lock on file {self.path}")
        lockfile.touch()

        try:

            if hasattr(self.path, "read"):
                self.messages = self.path.read()
            else:
                with open(self.path) as opath:
                    self.messages = load(opath)

            for key in self.messages:
                if self.messages[key] == "":
                    self.messages[key] = None

            lockfile.rm(absent="ignore")
        except Exception as e:
            lockfile.rm(absent="ignore")
            raise e

    def reset_messages(self):
        """Reset messages to their default values specified in `ifra.messenger.Receiver.DEFAULT_MESSAGES`"""
        self.messages = copy(self.DEFAULT_MESSAGES)
        self.save()


class NodeEmitter(Emitter):
    """Overloads `ifra.messenger.Messenger`

    Messages sent by a node. Listened to by the aggregator

    Messages
    --------
    stopped: bool
        Set to True by `ifra.node.Node.watch` when node stopped.
    creating: bool
        True during `ifra.node.Node` initialisation.
    error: str
        Set by `ifra.node.Node` methods if error occures. Will contain any error message raised.
    fitting: bool
        True during `ifra.node.Node.fit`.
    running: bool
        True during `ifra.node.Node.watch`.
    """
    DEFAULT_MESSAGES = {"stopped": False, "creating": True, "error": None, "fitting": False, "running": False}

    def __init__(self, path):
        super().__init__(path)


class CentralEmitter(Emitter):
    """Overloads `ifra.messenger.Messenger`

    Messages sent by central server. Listened to by the aggregator

    Messages
    --------
    stopped: bool
        Set to True by `ifra.central_server.CentralServer.watch` when server stopped.
    creating: bool
        True during `ifra.central_server.CentralServer` initialisation.
    error: str
        Set by `ifra.central_server.CentralServer` methods if error occures. Will contain any error message raised.
    running: bool
        True during `ifra.central_server.CentralServer.watch`.
    """
    DEFAULT_MESSAGES = {"stopped": False, "creating": True, "error": None, "running": False}

    def __init__(self, path):
        super().__init__(path)


class AggregatorEmitter(Emitter):
    """Overloads `ifra.messenger.Messenger`

    Messages sent by central server. Listened to by the aggregator

    Messages
    --------
    stopped: bool
        Set to True by `ifra.aggregator.Aggregator.watch` when aggregator stopped.
    creating: bool
        True during `ifra.aggregator.Aggregator` initialisation.
    error: str
        Set by `ifra.aggregator.Aggregator` methods if error occures. Will contain any error message raised.
    running: bool
        True during `ifra.aggregator.Aggregator.watch`.
    aggregating: bool
        True during `ifra.aggregator.Aggregator.aggregation`.
    """
    DEFAULT_MESSAGES = {"stopped": False, "creating": True, "error": None, "running": False, "aggregating": False}


class FromCentralReceiver(Receiver):
    """Overloads `ifra.messenger.Receiver`

    To read central server's messages

    For messages, see `ifra.messenger.CentralEmitter.DEFAULT_MESSAGES`
    """
    DEFAULT_MESSAGES = CentralEmitter.DEFAULT_MESSAGES


class FromNodeReceiver(Receiver):
    """Overloads `ifra.messenger.Receiver`

    To read a node's messages

    For messages, see `ifra.messenger.NodeEmitter.DEFAULT_MESSAGES`
    """
    DEFAULT_MESSAGES = NodeEmitter.DEFAULT_MESSAGES


class FromAggregatorReceiver(Receiver):
    """Overloads `ifra.messenger.Receiver`

    To read a aggregator's messages

    For messages, see `ifra.messenger.AggregatorEmitter.DEFAULT_MESSAGES`
    """
    DEFAULT_MESSAGES = AggregatorEmitter.DEFAULT_MESSAGES
