"""Check frozen source and optional local data-root artifacts without loading weights."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(args):
    package = args.package.resolve()
    source_manifest = json.loads((package / 'manifests/source_snapshot.json').read_text())
    checks = dict(source_manifest['files'])
    final_lock = package / 'manifests/package_files.json'
    if final_lock.exists():
        checks.update(json.loads(final_lock.read_text())['sha256'])
    failures = []
    for relative, expected in checks.items():
        path = package / relative
        if not path.is_file() or sha(path) != expected:
            failures.append(relative)
    if args.data_root:
        lock = json.loads((package / 'manifests/training_artifacts.lock.json').read_text())
        for relative, expected in lock['sha256'].items():
            path = args.data_root / relative
            if not path.is_file() or sha(path) != expected:
                failures.append('data_root:' + relative)
    print(json.dumps(dict(passed=not failures, checked_package_files=len(checks),
                          artifact_root_checked=args.data_root is not None, failures=failures)))
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path)
    main(parser.parse_args())
