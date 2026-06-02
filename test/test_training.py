import sys
from pathlib import Path

from network1d.training import training


TEST_DATA = Path(__file__).resolve().parent / 'test_data'


def test_training_smoke(tmp_path):
    training(
        False,
        0,
        graphs_folder='graphs/',
        data_location=str(TEST_DATA),
        out_dir=str(tmp_path),
        cli_args=[
            '--epochs', '3',
            '--data-location', str(TEST_DATA),
            '--out-dir', str(tmp_path),
        ],
    )


if __name__ == "__main__":
    training(
        False,
        0,
        graphs_folder='graphs/',
        data_location=str(TEST_DATA),
        cli_args=sys.argv[1:],
    )
