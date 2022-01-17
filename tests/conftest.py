from pytest import fixture
from transparentpath import Path


@fixture()
def clean_simulator():
    root = Path("tests/outputs/simulator", fs="local")
    if not root.isdir():
        root.mkdir()
    dirs = [
        root / "node_0",
        root / "node_1",
        root / "node_2",
        root / "node_3",
    ]
    for adir in dirs:
        if not adir.isdir():
            adir.mkdir()
        else:
            for f in adir.ls():
                f.rm(ignore_kind=True)
    for f in root.glob("ruleset*"):
        f.rm()
    yield


@fixture()
def clean_real():
    output_root = Path("tests/outputs/real", fs="local")
    if not output_root.isdir():
        output_root.mkdir()
    output_dirs = [
        output_root / "node_0",
        output_root / "node_1",
        output_root / "node_2",
        output_root / "node_3",
        output_root / "node_alone",
    ]
    for adir in output_dirs:
        if not adir.isdir():
            adir.mkdir()
        else:
            for f in adir.ls():
                f.rm(ignore_kind=True)
    for f in output_root.glob("ruleset*"):
        f.rm()

    configs_root = Path("tests/data/real", fs="local")
    configs_files = [
        configs_root / "node_0" / "public_configs.json",
        configs_root / "node_1" / "public_configs.json",
        configs_root / "node_2" / "public_configs.json",
        configs_root / "node_3" / "public_configs.json",
    ]
    tmp_configs_files = [
        configs_root / "node_0" / "public_configs_tmp.json",
        configs_root / "node_1" / "public_configs_tmp.json",
        configs_root / "node_2" / "public_configs_tmp.json",
        configs_root / "node_3" / "public_configs_tmp.json",
    ]
    for afile, newfile in zip(configs_files, tmp_configs_files):
        afile.cp(newfile)

        data_root = Path("tests/data/real", fs="local")
        data_dirs = [
            data_root / "node_0",
            data_root / "node_1",
            data_root / "node_2",
            data_root / "node_3",
        ]
        for data_dir in data_dirs:
            for pdffile in data_dir.glob("*.pdf"):
                pdffile.rm()

    yield

    for afile, newfile in zip(configs_files, tmp_configs_files):
        newfile.mv(afile)
