"""Create/verify a local small-adapter directory bundle. No network or ML imports."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.adapter_archive import BASE, REVISION, create_bundle, verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['create', 'verify'])
    parser.add_argument('--directory', required=True, help='Bundle directory')
    parser.add_argument('--source', help='Exact adapter files or empty pretraining source directory')
    parser.add_argument('--kind', choices=['trained_adapter', 'pretraining_roundtrip'], default='trained_adapter')
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--data-sha256', required=True)
    parser.add_argument('--evaluation-sha256', required=True)
    parser.add_argument('--manifest-sha256', help='Externally pinned digest, mandatory for verify')
    args = parser.parse_args()
    identity = {'base_model': BASE, 'base_revision': REVISION,
                'source_commit': args.source_commit, 'data_sha256': args.data_sha256,
                'evaluation_sha256': args.evaluation_sha256}
    try:
        if args.operation == 'create':
            if not args.source or args.manifest_sha256:
                parser.error('create requires --source and does not accept --manifest-sha256')
            digest = create_bundle(args.source, args.directory, identity, kind=args.kind)
            value = verify_bundle(args.directory, identity, digest)
        else:
            if not args.manifest_sha256 or args.source:
                parser.error('verify requires --manifest-sha256 and does not accept --source')
            digest = args.manifest_sha256
            value = verify_bundle(args.directory, identity, digest)
    except (ValueError, OSError) as error:
        parser.exit(1, f'Archive rejected: {error}\n')
    print(json.dumps({'manifest_sha256': digest, 'kind': value['kind'],
                      'contains_adapter': value['contains_adapter'],
                      'quality_qualified': False, 'files': value['files']}, sort_keys=True))


if __name__ == '__main__':
    main()
