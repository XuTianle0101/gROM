import argparse
from network1d.pipeline import EvalPipeline


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained VascGraphROM model')
    parser.add_argument('--config', type=str, required=False, help='Reserved for future consistency')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to trained_gnn.pms')
    args = parser.parse_args()
    model_dir = args.ckpt.rsplit('/', 1)[0]
    EvalPipeline().run(model_dir)


if __name__ == '__main__':
    main()
