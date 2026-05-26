import argparse
from network1d.config import TrainConfig
from network1d.pipeline import TrainPipeline


def main():
    parser = argparse.ArgumentParser(description='Train VascGraphROM model')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = TrainConfig(**__import__('yaml').safe_load(open(args.config, 'r')))
    TrainPipeline(cfg).run()


if __name__ == '__main__':
    main()
