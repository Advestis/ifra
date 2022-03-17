import pandas as pd

from ifra import Node
from ifra.configs import Config, NodeLearningConfig, NodeDataConfig
from ifra.fitters import Fitter, DecisionTreeClassificationFitter
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
    assert isinstance(node.fitter, (Fitter, DecisionTreeClassificationFitter))
    assert isinstance(node.updater, (NodeModelUpdater, AdaBoostNodeModelUpdater))
    assert isinstance(node.dataprep, (DataPrep, BinFeaturesDataPrep))
    assert node.datapreped is False
    assert node.data.dataprep_kwargs == node.learning_configs.dataprep_kwargs
    assert node.splitted is False
    assert node.model is None
    assert node.last_fetch is None
    assert not node.learning_configs.node_models_path.is_file()

    assert not (node.data.x_path.parent / "x_datapreped.csv").is_file()
    assert not (node.data.y_path.parent / "y_datapreped.csv").is_file()
    assert not (node.data.x_path.parent / "x_train_0.csv").is_file()
    assert not (node.data.y_path.parent / "y_train_0.csv").is_file()
    assert not (node.data.x_path.parent / "x_test.csv").is_file()
    assert not (node.data.y_path.parent / "y_test.csv").is_file()
    assert not (node.data.x_path.parent / "plots").is_dir()
    assert not (node.data.x_path.parent / "plots_datapreped").is_dir()
    assert not (node.data.x_path.parent / "plots_train_0").is_dir()

    assert node.emitter.doing is None
    assert node.emitter.error is None

    node.run(timeout=1, sleeptime=0.1)

    assert node.splitted
    assert node.datapreped
    assert node.data.x_datapreped_path.is_file()
    assert node.data.x_datapreped_path == node.data.x_path.parent / "x_datapreped.csv"
    assert node.data.y_datapreped_path.is_file()
    assert node.data.y_datapreped_path == node.data.y_path.parent / "y_datapreped.csv"
    assert node.data.x_train_path.is_file()
    assert node.data.x_train_path == node.data.x_path.parent / "x_train_0.csv"
    assert node.data.y_train_path.is_file()
    assert node.data.y_train_path == node.data.y_path.parent / "y_train_0.csv"
    assert node.data.x_train_path == node.data.x_test_path
    assert node.data.y_train_path == node.data.y_test_path
    assert (node.data.x_path.parent / "plots").is_dir()
    assert (node.data.x_path.parent / "bins.json").is_file()
    assert (node.data.x_path.parent / "plots_datapreped").is_dir()
    assert (node.data.x_path.parent / "plots_train_0").is_dir()
    assert (node.learning_configs.node_models_path / f"model_main_{node.filenumber}.csv").is_file()
    assert (node.learning_configs.node_models_path / f"model_{node.filenumber}_0.csv").is_file()

    try:
        pd.testing.assert_frame_equal(node.data.x_path.read(), node.data.x_datapreped_path.read())
    except AssertionError:
        pass
    else:
        raise AssertionError("x and x_datapreped DataFrames should be different")

    pd.testing.assert_frame_equal(node.data.y_path.read(), node.data.y_datapreped_path.read())

    assert node.data.x_datapreped_path.read().values.dtype == int
    assert node.data.x_train_path.read().values.dtype == int
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

    assert node.data.x_datapreped_path.read().values.dtype != int
    assert node.data.x_train_path.read().values.dtype != int
    assert node.data.x_path.read().values.dtype != int


def test_init_and_run_central_model_present(clean):
    node = Node(
        learning_configs=NodeLearningConfig("tests/data/node_test_with_central_model/learning_configs.json"),
        data=NodeDataConfig("tests/data/node_test_with_central_model/data_configs.json"),
    )
    node.run(timeout=1, sleeptime=0.1)

    x = node.data.x_path.read()
    x_train = node.data.x_train_path.read()
    assert len(x.index) > len(x_train.index)
