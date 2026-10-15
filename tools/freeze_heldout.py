"""Explicit pre-inference H14 source/spec freeze. Never use after heldout inference."""
import argparse
from datetime import datetime, timezone
from evaluation.heldout_fixtures import ROOT, FROZEN, MANIFEST, author_cases, canonical, digest, validate

FILES = ('evaluation/heldout_protocol.py', 'evaluation/heldout_fixtures.py',
         'evaluation/heldout_runner.py', 'evaluation/heldout/tasks.json',
         'docs/H14_PROTOCOL.md', 'docs/H14_RUN.md', 'specs/H14-heldout-frontier.md',
         'tools/freeze_heldout.py')
FILES += ('training/checkpoints.py', 'training/artifacts.py')


def freeze(replace=False):
    if MANIFEST.exists() and not replace:
        raise ValueError('Existing freeze preserved; explicit pre-inference replacement required')
    FROZEN.parent.mkdir(exist_ok=True)
    FROZEN.write_bytes(canonical(validate(author_cases())))
    manifest = {'schema': 'hala.h14.freeze.v1', 'phase': 'P2_before_inference',
                'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
                'instances': 64, 'families': 8, 'languages': {'en': 32, 'ar': 32},
                'files': {relative: digest(ROOT / relative) for relative in FILES}}
    MANIFEST.write_bytes(canonical(manifest))
    print('manifest_sha256=' + digest(MANIFEST))
    print('dataset_sha256=' + digest(FROZEN))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replace-pre-inference', action='store_true')
    args = parser.parse_args()
    freeze(args.replace_pre_inference)
