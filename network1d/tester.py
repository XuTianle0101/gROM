"""Legacy compatibility wrapper for refactored evaluation module."""
from network1d.eval import *

if __name__ == "__main__":
    import sys
    from network1d.eval import get_dataset_and_gnn, evaluate_all_models
    import os, shutil
    path = sys.argv[1]
    dataset, gnn_model, params = get_dataset_and_gnn(path)
    if os.path.exists("results"):
        shutil.rmtree("results")
    evaluate_all_models(dataset, "train", gnn_model, params, False)
    evaluate_all_models(dataset, "test", gnn_model, params, False)
