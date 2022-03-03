from copy import deepcopy, copy
from pathlib import Path
from time import time, sleep
from typing import Union
from json import load
from transparentpath import TransparentPath
import logging

logger = logging.getLogger(__name__)


def get_lock(path: TransparentPath, timeout: int = 5, exists: str = "wait") -> TransparentPath:
    lockfile = path.append(".lock")
    if not lockfile.parent.is_dir():
        lockfile.parent.mkdir(parents=True)
        lockfile.touch()
        return lockfile

    t = time()
    while lockfile.is_file():
        if exists != "wait":
            raise ValueError(
                f"Lock file '{lockfile}' exists. "
                "This can indicate a crash of a previous execution, or the fact that two"
                "sessions are running at the same time. Delete this file and restart."
            )
        sleep(0.5)
        if time() - t > timeout:
            raise FileExistsError(f"File '{lockfile}' exists : could not lock on corresponding file.")
    lockfile.touch()
    return lockfile


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
        `ifra.messenger.Emitter.DEFAULT_MESSAGES`. If it exists and it indicates that the corresponding object is
        doint something, will raise an error. It it exists and says taht the corresponding object is doing nothing,
        overwrites the file.
     messages: dict
        A dictionnary of messages

    Messages
    --------
    error: str
        Set by the emitting object error occures. Will contain any error message raised.
    doing: str
        Name of the function currently being run by the emitting object
    """
    DEFAULT_MESSAGES = {"error": None, "doing": None}
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

        lockfile = get_lock(path, exists="raise")

        self.path = path
        if self.path.is_file():
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
        else:
            self.messages = {"doing": None, "error": None}

        if self.doing is not None:
            raise ValueError(
                f"Message file '{path}' existed at Emitter creation and had 'doing={self.doing}'. "
                f"This can indicate a crash of a previous execution, or the fact that two"
                f"sessions are running at the same time. Delete this file and restart."
            )

        lockfile.rm(absent="ignore")
        self.reset_messages()

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

    def send(self, **kwargs):
        """Sends several messages at once. Keys in kwargs must be present in `ifra.messenger.Emitter.DEFAULT_MESSAGES`
        """
        for key in kwargs:
            if key not in self.messages and key not in self.__class__.DEFAULT_MESSAGES:
                raise ValueError(f"Message named '{key}' is not allowed")
            self.messages[key] = kwargs[key]
        self.save()

    def save(self) -> None:
        """Saves the current messages into the file it used to load."""
        to_save = copy(self.messages)

        lockfile = get_lock(self.path)

        try:
            for key in to_save:
                if key not in self.__class__.DEFAULT_MESSAGES:
                    raise ValueError(f"Forbidden message '{key}'")
                if to_save[key] is None:
                    # noinspection PyTypeChecker
                    to_save[key] = ""

            self.path.write(to_save)

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
        lockfile = get_lock(self.path)
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

    def __init__(self, path: Union[str, Path, TransparentPath], wait: bool = True):
        """
        Parameters
        ----------
        path: Union[str, Path, TransparentPath]
            The path to the json file containing the messages. Will be casted into a transparentpath if a str is
            given, so make sure that if you pass a str that should be local, you did not set a global file system.
        wait: bool
            If True, will wait for 'path' to point to an existing file for `ifra.messenger.Emitter.DEFAULT_MESSAGES`
            seconds at Receiver creation if it does not exist. Else, messages will be those in
            `ifra.messenger.Emitter.DEFAULT_MESSAGES`
        """

        if type(path) == str:
            path = TransparentPath(path)

        if not path.suffix == ".json":
            raise ValueError("Messenger path must be a json")
        self.path = path
        self.messages = {}
        self.new = True
        self.get_latest_messages(wait)

    def __getattr__(self, item):
        if item == "messages":
            raise ValueError("Messenger object's 'messages' not set")
        if item == "path":
            raise ValueError("Messenger object's 'path' not set")
        if item not in self.messages:
            raise ValueError(f"No messages named '{item}' was found")
        return self.messages[item]

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        setattr(result, "messages", deepcopy(self.messages, memo))
        return result

    def get_latest_messages(self, wait: bool = False) -> bool:
        """Gets latest messages by reading the messages json file. If file does not exist and it is the first time we
        read it (object init), wait for it for at most `ifra.messenger.Receiver.timeout_when_missing` seconds then
        raises TimeoutError. If file does not exist and it is not the first time we read it, returns False. If
        messages are read, returns True"""
        first_message = True

        if self.__class__.timeout_when_missing is None:
            raise ValueError("'timeout_when_missing' can not be None")

        t = time()
        while not self.path.is_file() and time() - t < self.timeout_when_missing:
            if not self.new:
                logger.warning(f"Message file {self.path} disapread. No message received.")
                return False
            if not wait:
                logger.warning(
                    f"Message file {self.path} does not exist. No message received, using defaults messages."
                )
                self.messages = copy(self.DEFAULT_MESSAGES)
                return False
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

        lockfile = get_lock(self.path)

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
        return True

    def reset_messages(self):
        """Reset messages to their default values specified in `ifra.messenger.Receiver.DEFAULT_MESSAGES`"""
        self.messages = copy(self.DEFAULT_MESSAGES)
        self.save()
