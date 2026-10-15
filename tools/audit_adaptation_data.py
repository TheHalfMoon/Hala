"""Reproduce H18 readiness evidence; exit 2 means valid evidence but blocked training."""
import argparse
import json
from pathlib import Path

from training.adaptation import audit, read
from training.artifacts import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = read(args.root / 'experiments/adaptation/shared-multitask.json')
    result = audit(args.root, config)
    result['evidence_sha256']['experiments/adaptation/shared-multitask.json'] = sha256_file(
        args.root / 'experiments/adaptation/shared-multitask.json')
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(payload, encoding='utf-8')
    else:
        print(payload, end='')
    return 0 if result['training_ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
