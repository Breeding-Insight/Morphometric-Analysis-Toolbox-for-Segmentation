"""Launch the Streamlit GUI via `mats app`."""

import subprocess
import sys
from pathlib import Path


def launch(extra_args=None):
    """Run `streamlit run Home.py`, forwarding any extra args verbatim.

    Extra args are passed straight through to Streamlit, so the Open OnDemand
    launcher can supply `--server.address`, `--server.port`,
    `--server.baseUrlPath`, etc.
    """
    entry = Path(__file__).with_name("Home.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(entry)]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(launch(sys.argv[1:]))
