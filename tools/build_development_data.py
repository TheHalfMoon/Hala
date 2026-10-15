"""Build a new immutable development bundle; never train or load a model."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from development.freeze import freeze_bundle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--font', type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze_bundle(args.output, json.loads(args.settings.read_text(encoding='utf-8')), args.font), sort_keys=True))
