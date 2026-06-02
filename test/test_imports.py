import subprocess
import sys


def test_package_imports():
    import graph1d.generate_dataset  # noqa: F401
    import graph1d.generate_normalized_graphs  # noqa: F401
    import network1d.meshgraphnet  # noqa: F401
    import network1d.rollout  # noqa: F401
    import network1d.tester  # noqa: F401
    import network1d.training  # noqa: F401
    import tools.io_utils  # noqa: F401


def test_imports_from_non_repo_directory(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import network1d.training; import network1d.tester',
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
