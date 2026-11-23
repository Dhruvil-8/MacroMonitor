import sys
from pathlib import Path

# Add pipeline/src to sys.path
sys.path.insert(0, str(Path("pipeline/src").resolve()))

from radar.cli import main

if __name__ == "__main__":
    ret = main(["export", "--target", "local"])
    print(f"Export returned: {ret}")
