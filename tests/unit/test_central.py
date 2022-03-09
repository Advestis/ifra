from ifra.central_server import CentralServer
from ifra.configs import CentralConfig, Config


def test_simple_init_and_run():
    server = CentralServer(central_configs=CentralConfig("tests/data/central_configs.json"))
    assert isinstance(server.central_configs, (Config, CentralConfig))

    assert server.emitter.doing is None
    assert server.emitter.error is None

    server.run(timeout=1, sleeptime=0.1)

    assert server.emitter.doing is None
    assert server.emitter.error is None

    server = CentralServer(central_configs=CentralConfig("tests/data/central_configs.json"))
    assert isinstance(server.central_configs, (Config, CentralConfig))
    server.run(timeout=1, sleeptime=0.1)
