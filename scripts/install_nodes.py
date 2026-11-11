"""Install the final RGB240 package without overwriting an existing different copy."""
import argparse
import os
from pathlib import Path
import shutil


def install(comfy_root):
    repo = Path(__file__).resolve().parents[1]
    source = repo / 'custom_nodes/product_studio_rgb240'
    target = comfy_root / 'custom_nodes/product_studio_rgb240'
    if target.exists():
        a = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        b = {str(p.relative_to(target)): p.read_bytes() for p in target.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        if a != b:
            raise RuntimeError(f'Back up the existing different node folder first: {target}')
    else:
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__'))
    workflow = comfy_root / 'user/default/workflows/RGB240.json'
    workflow.parent.mkdir(parents=True, exist_ok=True)
    contents = (repo / 'workflows/rgb240_ui.json').read_bytes()
    if workflow.exists() and workflow.read_bytes() != contents:
        raise RuntimeError(f'Existing workflow was not overwritten: {workflow}')
    workflow.write_bytes(contents)
    # Configure a private, per-install Ultralytics cache. A fresh ComfyUI process
    # then resolves WEIGHTS_DIR/clip to the locked model path, not a Mac home path.
    os.environ['YOLO_CONFIG_DIR'] = str(comfy_root / '.rgb240-ultralytics')
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    from ultralytics import settings
    settings.update({'weights_dir': str(comfy_root / 'models/detection')})
    print(f'Installed {target}')
    print('Start through scripts/start_reference.sh so the same cache path is used.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--comfy-root', type=Path, required=True)
    install(parser.parse_args().comfy_root.resolve())
