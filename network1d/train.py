from network1d.config import TrainConfig
from network1d.pipeline import TrainPipeline


if __name__ == '__main__':
    config = TrainConfig.from_args()
    TrainPipeline(config).run()
