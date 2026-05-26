import argparse
import torch as th
from network1d.tester import get_dataset_and_gnn
from network1d.rollout import rollout


def main():
    parser = argparse.ArgumentParser(description='Run rollout evaluation on test split')
    parser.add_argument('--config', type=str, required=False, help='Reserved for future consistency')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to trained_gnn.pms')
    args = parser.parse_args()

    model_dir = args.ckpt.rsplit('/', 1)[0]
    dataset, gnn_model, params = get_dataset_and_gnn(model_dir)
    print('==========test rollout==========')
    for i, graph in enumerate(dataset['test'].graphs):
        with th.no_grad():
            _, errs_n, errs, _, elapsed = rollout(gnn_model, params, graph)
        print(f"{dataset['test'].graph_names[i]}: norm={errs_n}, phys={errs}, time={elapsed:.3f}s")


if __name__ == '__main__':
    main()
