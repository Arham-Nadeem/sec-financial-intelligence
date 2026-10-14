#Convenience launcher that starts the Streamlit research interface.
from __future__ import annotations

import subprocess
import sys


if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "src/sec_intelligence/ui/app.py"],
        check=True,
    )
