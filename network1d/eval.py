"""Refactored evaluation entrypoint.

Delegates to the legacy script to preserve exact functionality.
"""

import runpy


if __name__ == "__main__":
    runpy.run_module("network1d.tester", run_name="__main__")
