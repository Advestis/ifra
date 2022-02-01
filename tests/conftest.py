from pytest import fixture
from transparentpath import Path


@fixture()
def clean():
    output_root = Path("tests/outputs", fs="local")
    if not output_root.is_dir():
        output_root.mkdir()

    output_dirs = [
        output_root / "node_0",
        output_root / "node_1",
        output_root / "node_2",
        output_root / "node_test",
        output_root / "central_server",
        output_root / "aggregator",
    ]
    for adir in output_dirs:
        if not adir.is_dir():
            adir.mkdir()
        else:
            for f in adir.ls():
                if "message" not in str(f):
                    f.rm(ignore_kind=True)
    for f in output_root.glob("ruleset*"):
        f.rm()

    data_root = Path("tests/data", fs="local")
    data_dirs = [
        data_root / "node_0",
        data_root / "node_1",
        data_root / "node_2",
        data_root / "node_test",
    ]
    for data_dir in data_dirs:
        (data_dir / "plots").rmdir(absent="ignore")
        (data_dir / "plots_datapreped").rmdir(absent="ignore")

    (data_root / "node_test" / "x_to_use.csv").rm(absent="ignore")
    (data_root / "node_test" / "y_to_use.csv").rm(absent="ignore")

    yield

    (data_root / "node_test" / "x_to_use.csv").rm(absent="ignore")
    (data_root / "node_test" / "y_to_use.csv").rm(absent="ignore")
    (data_root / "node_test" / "plots").rmdir(absent="ignore")
    (data_root / "node_test" / "plots_datapreped").rmdir(absent="ignore")
    (output_root / "node_test").rmdir(absent="ignore")
