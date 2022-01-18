from copy import deepcopy, copy
from pathlib import Path
from time import time, sleep
from typing import Optional, Union
from json import load
from transparentpath import TransparentPath


class NodeMessenger:
    """Instance to parse messages from central server to Nodes and vice versa.

     Uses the same logic than `ifra.configs.Config` to handle the json file that is read or written by
     `ifra.node.Node` and `ifra.central_server.NodeGate`.
     
     Only messages which names are in `ifra.messenger.NodeMessenger.DEFAULT_MESSAGES` are allowed.

     Attributes
     ----------
     path: TransparentPath
        The path to the json file containing the messages. Will be casted into a transparentpath if a str is
        given, so make sure that if you pass a string that should be local, you did not set a global file system.
        If it does not exist, it is created with the default messages present in
        `ifra.messenger.NodeMessenger.DEFAULT_MESSAGES`
     messages: dict
        A dictionnary of messages

    stop: bool
        Set to True by `ifra.central_server.CentralServer` when the learning is over.
    error: str
        Set by `ifra.node.Node.watch`. Will contain any error that `ifra.node.Node.watch` have raised
    central_error: str
        Set by `ifra.central_server.CentralServer.watch`. Will contain any error that
        `ifra.central_server.CentralServer.watch` have raised
    fitting: bool
        Set by `ifra.node.Node.fit`. True if the node is currently fitting.
    running: bool
        Set by `ifra.node.Node.watch`. True if the node is currently running.
     """
    DEFAULT_MESSAGES = {"stop": False, "error": None, "central_error": None, "fitting": False, "running": False}

    def __init__(self, path: Optional[Union[str, Path, TransparentPath]] = None):
        """
        Parameters
        ----------
        path: Optional[Union[str, Path, TransparentPath]]
            The path to the json file containing the messages. Will be casted into a transparentpath if a str is
            given, so make sure that if you pass a str that should be local, you did not set a global file system.
        """

        if type(path) == str:
            path = TransparentPath(path)

        if not path.suffix == ".json":
            raise ValueError("Messenger path must be a json")

        lockfile = path.append(".locked")
        t = time()
        while lockfile.is_file():
            sleep(0.5)
            if time() - t > 5:
                raise FileExistsError(f"File {lockfile} exists : could not lock on file {path}")
        lockfile.touch()

        try:

            self.path = path

            if not path.is_file():
                lockfile.rm(absent="ignore")
                self.reset_messages()
                return

            if hasattr(path, "read"):
                self.messages = path.read()
            else:
                with open(path) as opath:
                    self.messages = load(opath)

            for key in self.messages:
                if self.messages[key] == "":
                    self.messages[key] = None
                    
            modified = False
            for key in self.DEFAULT_MESSAGES:
                if key not in self.messages:
                    modified = True
                    self.messages[key] = self.DEFAULT_MESSAGES[key]
            if modified:
                self.save()

            lockfile.rm(absent="ignore")
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
        if item not in self.messages and item not in self.DEFAULT_MESSAGES:
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
                if key not in self.DEFAULT_MESSAGES:
                    raise ValueError(f"Forbidden message '{key}'")
                if to_save[key] is None:
                    to_save[key] = ""

            self.path.write(to_save)
            lockfile.rm(absent="ignore")
        except Exception as e:
            lockfile.rm(absent="ignore")
            raise e

    def get_latest_messages(self):
        """Gets latest messages by reading the messages json file again. File must exist."""
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
        """Reset messages to their default values specified in `ifra.messenger.NodeMessenger.DEFAULT_MESSAGES`"""
        self.messages = copy(self.DEFAULT_MESSAGES)
        self.save()

    def rm(self):
        """Removes the NodeMessenger's json file."""
        self.path.rm(absent="ignore")
