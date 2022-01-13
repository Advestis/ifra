from ifra import CentralServer, Node
import numpy as np
from bisect import bisect


def dataprep_method(x, y):
    def find_bins(xx, nbins: int):
        q_list = np.arange(100.0 / nbins, 100.0, 100.0 / nbins)
        bins = np.array([np.nanpercentile(xx, i) for i in q_list])
        return bins

    def get_bins(xx, nb_bucket: int):
        bins = find_bins(xx, nb_bucket)
        while len(set(bins.round(5))) != len(bins):
            nb_bucket -= 1
            bins = find_bins(xx, nb_bucket)
        if len(bins) != nb_bucket - 1:
            raise ValueError(f"Error in get_bins : {len(bins) + 1} bins where found but {nb_bucket} were asked.")
        return bins

    def dicretize(x_series):
        bins = get_bins(x_series, 5)
        mask = np.isnan(x_series)
        discrete_x = x_series.apply(lambda var: bisect(bins, var))
        discrete_x[mask] = np.nan
        return discrete_x

    x = x.apply(lambda xx: dicretize(xx), axis=0)
    return x, y


def test_iris(clean):
    nodes = [
        Node(
            public_configs_path="tests/data/learning_real.json",
            path_configs_path=f"tests/data/node_{i}/path_configs.json",
            dataprep_method=dataprep_method,
        )
        for i in range(4)
    ]
    cs = CentralServer(nodes=nodes)
    cs.fit(5, save_path="tests/outputs")


# def test_iris_one_iteration_one_node():
#     nodes = [
#         Node(
#             public_configs_path="tests/data/learning_real.json",
#             path_configs_path="tests/data/node_alone/path_configs.json",
#             dataprep_method=dataprep_method,
#         )
#     ]
#     cs = CentralServer(nodes=nodes)
#     cs.fit(1, save_path="tests/outputs/alone")
