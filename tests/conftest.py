from pytest import fixture
from transparentpath import Path


@fixture()
def clean():
    root = Path("tests/outputs", fs="local")
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
