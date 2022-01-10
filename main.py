import os
from infra import CentralServer
import logging
import adutils
adutils.init("logger")
from adutils import setup_logger
setup_logger()

logger = logging.getLogger(__name__)
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/pcotte/second-capsule-253207-72efd01e4e7f.json"


if __name__ == "__main__":
    cs = CentralServer(learning_configs_path="tests/data/learning.json")
