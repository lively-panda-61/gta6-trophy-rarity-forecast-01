"""Entrypoint: read every <id>.in in the input directory, write one
<id>.out per case to the output directory. Standard library only.

The grader runs `python3 -P -s /bot/main.py` with MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR unset (so /input and /output); local runs set both.
"""
import json
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so
# `import model` finds model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import forecast_case  # noqa: E402

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case_files = sorted(INPUT_DIR.glob("*.in"))
    print("[forecast] input dir: %s (%d case file(s))"
          % (INPUT_DIR, len(case_files)))
    for case_file in case_files:
        with open(case_file, encoding="utf-8") as fh:
            case = json.load(fh)
        out = forecast_case(case)
        out_path = OUTPUT_DIR / (case_file.stem + ".out")
        raw = json.dumps(out, separators=(",", ":")).encode("utf-8")
        print("[forecast] %s: %d trophies, %d bytes -> %s"
              % (case.get("id"), len(out["trophies"]), len(raw), out_path))
        if len(raw) > 65536:
            raise SystemExit("[forecast] output over 65,536 bytes: %s"
                             % case_file.stem)
        with open(out_path, "wb") as fh:
            fh.write(raw)
    print("[forecast] done")


if __name__ == "__main__":
    main()
