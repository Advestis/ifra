from ifra import Node
from ifra.configs import Config, NodeLearningConfig, NodeDataConfig
from ifra.fitters import Fitter, DecisionTreeFitter
from ifra.node_model_updaters import NodeModelUpdater, AdaBoostNodeModelUpdater
from ifra.datapreps import DataPrep, BinFeaturesDataPrep
from ..alternates.dataprep import AlternateDataPrep
from ..alternates.updater import AlternateUpdater
from ..alternates.fitter import AlternateFitter


def test_init_and_run_simple(clean):
    node = Node(
        learning_configs=NodeLearningConfig("tests/data/node_test/learning_configs.json"),
        data=NodeDataConfig("tests/data/node_test/data_configs.json"),
    )
    assert isinstance(node.learning_configs, (Config, NodeLearningConfig))
    assert isinstance(node.data, (Config, NodeDataConfig))
    assert isinstance(node.fitter, (Fitter, DecisionTreeFitter))
    assert isinstance(node.updater, (NodeModelUpdater, AdaBoostNodeModelUpdater))
    assert isinstance(node.dataprep, (DataPrep, BinFeaturesDataPrep))
    assert node.datapreped is False
    assert node.data.dataprep_kwargs == node.learning_configs.dataprep_kwargs
    assert node.copied is False
    assert node.model is None
    assert node.last_fetch is None
    assert not node.learning_configs.node_models_path.is_file()

    assert not (node.data.x_path.parent / "x_to_use.csv").is_file()
    assert not (node.data.y_path.parent / "y_to_use.csv").is_file()
    assert not (node.data.x_path.parent / "plots").is_dir()
    assert not (node.data.x_path.parent / "plots_datapreped").is_dir()

    assert node.emitter.doing is None
    assert node.emitter.error is None

    node.run(timeout=1, sleeptime=0.1)

    assert node.copied
    assert node.datapreped
    assert (node.data.x_path.parent / "x_to_use.csv").is_file()
    assert (node.data.y_path.parent / "y_to_use.csv").is_file()
    assert (node.data.x_path.parent / "plots").is_dir()
    assert (node.data.x_path.parent / "plots_datapreped").is_dir()
    assert node.learning_configs.node_models_path.is_file()

    assert node.data.x_path_to_use.read().values.dtype == int
    assert node.data.x_path.read().values.dtype != int

    assert node.emitter.doing is None
    assert node.emitter.error is None


def test_init_alternate_dataprep_updater_fitter(clean):
    node = Node(
        learning_configs=NodeLearningConfig("tests/data/node_test/alternate_learning_configs.json"),
        data=NodeDataConfig("tests/data/node_test/data_configs.json"),
    )
    assert isinstance(node.dataprep, AlternateDataPrep)
    assert isinstance(node.updater, AlternateUpdater)
    assert isinstance(node.fitter, AlternateFitter)

    node.run(timeout=1, sleeptime=0.1)

    assert node.data.x_path_to_use.read().values.dtype != int
    assert node.data.x_path.read().values.dtype != int
