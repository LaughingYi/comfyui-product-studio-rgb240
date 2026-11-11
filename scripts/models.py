"""Download or verify only the five locked RGB240 model files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def verify(root, download=False):
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'models/manifest.json').read_text())
    for item in manifest['files']:
        target = (root / item['target']).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError('Model target escapes model root')
        if not target.exists() and download:
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.download')
            if not item['url'].startswith('https://'):
                raise ValueError('Only HTTPS model sources are allowed')
            subprocess.run(['curl', '--fail', '--location', '--retry', '3',
                            '--output', str(partial), '--url', item['url']], check=True)
            if partial.stat().st_size != item['bytes'] or digest(partial) != item['sha256']:
                raise RuntimeError(f"Downloaded file failed verification: {item['id']}")
            partial.replace(target)
        if not target.is_file():
            raise FileNotFoundError(f'Missing {target}; rerun with --download or transfer the locked file')
        if target.stat().st_size != item['bytes'] or digest(target) != item['sha256']:
            raise RuntimeError(f'Existing file differs from the locked model; not overwritten: {target}')
        print(f"OK {item['target']}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    verify(args.model_root, args.download)
