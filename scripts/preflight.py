import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess


def main(root):
    repo = Path(__file__).resolve().parents[1]
    lock = json.loads((repo / 'SOURCE_LOCK.json').read_text())
    for folder, expected in [(root, lock['comfyui']['commit']),
                             (root / 'custom_nodes/ComfyUI_BiRefNet_ll', lock['birefnet_nodes']['commit'])]:
        actual = subprocess.check_output(['git', '-C', str(folder), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != expected:
            raise RuntimeError(f'Unvalidated source revision: {folder}: {actual}')
    os.environ['YOLO_CONFIG_DIR'] = str(root / '.rgb240-ultralytics')
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    import torch
    from ultralytics.utils import WEIGHTS_DIR
    if WEIGHTS_DIR.resolve() != (root / 'models/detection').resolve():
        raise RuntimeError('CLIP cache path is wrong; run install_nodes.py first')
    packages = {}
    for name in ['torch', 'torchvision', 'numpy', 'opencv-python', 'Pillow', 'transformers',
                 'timm', 'ultralytics', 'safetensors', 'clip']:
        packages[name] = importlib.metadata.version(name)
    print(json.dumps({'os': platform.platform(), 'python': platform.python_version(),
                      'cuda_available': torch.cuda.is_available(),
                      'cuda_runtime': torch.version.cuda,
                      'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      'packages': packages, 'reference_mode': 'CPU, not GPU accelerated'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--comfy-root', type=Path, required=True)
    main(parser.parse_args().comfy_root.resolve())
