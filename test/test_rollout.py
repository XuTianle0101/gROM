import math
from pathlib import Path

from network1d.rollout import rollout
from network1d.tester import get_gnn_and_graphs


TEST_DATA = Path(__file__).resolve().parent / 'test_data'


def test_rollout_regression():
    path = TEST_DATA / 'gnn_model'
    graphs_folder = 'graphs/'
    gnn_model, graphs, params = get_gnn_and_graphs(path, graphs_folder,
                                                   str(TEST_DATA))
    graph_name = 's0095_0001.0.3.grph'
    _, _, err, _, _ = rollout(gnn_model, params, graphs[graph_name])

    tol = 1e-4
    assert math.isclose(float(err[0]), 0.00617872, rel_tol=tol)
    assert math.isclose(float(err[1]), 0.01505195, rel_tol=tol)


if __name__ == "__main__":
    test_rollout_regression()
