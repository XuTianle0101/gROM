import argparse
from network1d.pipeline import EvalPipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    args = parser.parse_args()
    EvalPipeline().run(args.model_path)
