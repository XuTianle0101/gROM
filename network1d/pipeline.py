"""New training/evaluation framework pipeline preserving legacy behavior."""

import torch as th
import torch.distributed as dist
import tools.io_utils as io
import network1d.training as legacy_training
import network1d.tester as legacy_tester
from network1d.config import TrainConfig


class TrainPipeline:
    def __init__(self, config: TrainConfig):
        self.config = config

    def run(self):
        rank = 0
        try:
            parallel = True
            dist.init_process_group(backend='mpi')
            rank = dist.get_rank()
            print("my rank = %d, world = %d." % (rank, dist.get_world_size()), flush=True)
            th.backends.cudnn.enabled = False
        except RuntimeError:
            parallel = False
            print("MPI not supported. Running serially.")

        legacy_training.parse_command_line_arguments = lambda: (self.config.to_legacy_params(), self.config)
        types_to_keep = ['synthetic_aorta_coarctation', 'synthetic_pulmonary', 'synthetic_aortofemoral']
        nodes_features = ['area', 'tangent', 'type', 'T', 'dip', 'sysp', 'resistance1', 'capacitance', 'resistance2']
        edges_features = ['rel_position', 'distance']
        features = {'nodes_features': nodes_features, 'edges_features': edges_features}
        legacy_training.training(parallel, rank, data_location=io.data_location(), types_to_keep=types_to_keep, features=features)


class EvalPipeline:
    def run(self, model_path: str):
        dataset, gnn_model, params = legacy_tester.get_dataset_and_gnn(model_path)
        legacy_tester.evaluate_all_models(dataset, 'train', gnn_model, params, False)
        legacy_tester.evaluate_all_models(dataset, 'test', gnn_model, params, False)
