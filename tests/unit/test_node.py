from ifra import Node
from ifra.configs import Config, NodePublicConfig, NodeDataConfig
from ifra.messenger import NodeMessenger
from ifra.fitters import Fitter, DecisionTreeFitter
from ifra.updaters import Updater, AdaBoostUpdater
from ifra.datapreps import DataPrep, BinFeaturesDataPrep
from ..alternates.dataprep import AlternateDataPrep
from ..alternates.updater import AlternateUpdater
from ..alternates.fitter import AlternateFitter


def test_init_and_watch_simple(clean):
    node = Node("tests/data/node_test/public_configs.json", "tests/data/node_test/data_configs.json")
    assert isinstance(node.public_configs, (Config, NodePublicConfig))
    assert isinstance(node.data, (Config, NodeDataConfig))
    assert isinstance(node.messenger, NodeMessenger)
    assert node.messenger.running is False
    assert node.messenger.fitting is False
    assert node.messenger.stop is False
    assert node.messenger.error is None
    assert node.messenger.central_error is None
    assert isinstance(node.fitter, (Fitter, DecisionTreeFitter))
    assert isinstance(node.updater, (Updater, AdaBoostUpdater))
    assert isinstance(node.dataprep, (DataPrep, BinFeaturesDataPrep))
    assert node.datapreped is False
    assert node.data.dataprep_kwargs == node.public_configs.dataprep_kwargs
    assert node.copied is False
    assert node.ruleset is None
    assert node.last_fetch is None

    assert not (node.data.x.parent / "x_datapreped.csv").is_file()
    assert not (node.data.y.parent / "y_datapreped.csv").is_file()
    assert not (node.data.x.parent / "x_datapreped_copy_for_learning.csv").is_file()
    assert not (node.data.y.parent / "y_datapreped_copy_for_learning.csv").is_file()
    assert not (node.data.x.parent / "plots").is_dir()
    assert not (node.data.x.parent / "plots_datapreped").is_dir()

    node.watch(timeout=5, sleeptime=1)

    assert node.copied
    assert node.datapreped
    assert node.messenger.fitting is False
    assert node.messenger.stop is True
    assert (node.data.x.parent / "x_datapreped.csv").is_file()
    assert (node.data.y.parent / "y_datapreped.csv").is_file()
    assert (node.data.x.parent / "x_datapreped_copy_for_learning.csv").is_file()
    assert (node.data.y.parent / "y_datapreped_copy_for_learning.csv").is_file()
    assert (node.data.x.parent / "plots").is_dir()
    assert (node.data.x.parent / "plots_datapreped").is_dir()
    assert node.public_configs.local_model_path.is_file()


def test_init_alternate_dataprep_updater_fitter():
    node = Node("tests/data/node_test/alternate_public_configs.json", "tests/data/node_test/data_configs.json")
    assert isinstance(node.dataprep, AlternateDataPrep)
    assert isinstance(node.updater, AlternateUpdater)
    assert isinstance(node.fitter, AlternateFitter)
