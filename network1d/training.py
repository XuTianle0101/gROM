"""Legacy compatibility wrapper for refactored training module."""
from network1d.train import *

if __name__ == "__main__":
    from network1d.train import training
    import torch as th
    import torch.distributed as dist
    rank = 0
    try:
        parallel = True
        dist.init_process_group(backend="mpi")
        rank = dist.get_rank()
        print("my rank = %d, world = %d." % (rank, dist.get_world_size()), flush=True)
        th.backends.cudnn.enabled = False
    except RuntimeError:
        parallel = False
        print("MPI not supported. Running serially.")

    types_to_keep = ["synthetic_aorta_coarctation", "synthetic_pulmonary", "synthetic_aortofemoral"]
    nodes_features = ["area","tangent","type","T","dip","sysp","resistance1","capacitance","resistance2"]
    edges_features = ["rel_position","distance"]
    features = {"nodes_features": nodes_features, "edges_features": edges_features}
    training(parallel, rank, types_to_keep=types_to_keep, features=features)
