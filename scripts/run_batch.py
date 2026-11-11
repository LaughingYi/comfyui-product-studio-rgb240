"""Upload, queue and download images serially through ComfyUI's standard API."""
import argparse
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request
import uuid


def request(url, body=None, content_type=None):
    headers = {'Content-Type': content_type} if content_type else {}
    return urllib.request.urlopen(urllib.request.Request(url, data=body, headers=headers), timeout=60)


def run_one(image, destination, base, workflow, timeout):
    boundary = 'rgb240' + uuid.uuid4().hex
    filename = uuid.uuid4().hex + image.suffix.lower()
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            'Content-Type: application/octet-stream\r\n\r\n').encode()
    body += image.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
    with request(base + '/upload/image', body, 'multipart/form-data; boundary=' + boundary) as response:
        uploaded = json.load(response)
    graph = json.loads(workflow.read_text())
    graph['1']['inputs']['image'] = '/'.join(x for x in [uploaded.get('subfolder'), uploaded['name']] if x)
    graph['7']['inputs']['filename_prefix'] = 'rgb240/' + uuid.uuid4().hex
    payload = json.dumps({'prompt': graph, 'client_id': 'rgb240-' + uuid.uuid4().hex}).encode()
    with request(base + '/prompt', payload, 'application/json') as response:
        queued = json.load(response)
    if queued.get('node_errors'):
        raise RuntimeError(queued)
    prompt_id = queued['prompt_id']
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with request(base + '/history/' + prompt_id) as response:
            history = json.load(response).get(prompt_id)
        if history:
            if history.get('status', {}).get('status_str') != 'success':
                raise RuntimeError(json.dumps(history.get('status'), ensure_ascii=False))
            output = history['outputs']['7']['images'][0]
            query = urllib.parse.urlencode({k: output[k] for k in ['filename', 'subfolder', 'type']})
            with request(base + '/view?' + query) as response:
                pixels = response.read()
            if not pixels.startswith(b'\x89PNG\r\n\x1a\n'):
                raise RuntimeError('Server did not return a PNG')
            destination.write_bytes(pixels)
            print(json.dumps({'input': image.name, 'output': str(destination), 'prompt_id': prompt_id}, ensure_ascii=False), flush=True)
            return
        time.sleep(2)
    raise TimeoutError(f'Job is still on the server; inspect prompt_id={prompt_id}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8188')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    if urllib.parse.urlsplit(args.url).scheme not in ['http', 'https']:
        raise ValueError('Use an HTTP(S) ComfyUI URL')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp'])
    if not images:
        raise SystemExit('No supported input images found')
    workflow = Path(__file__).resolve().parents[1] / 'workflows/rgb240_api.json'
    for image in images:
        destination = args.output_dir / (image.name + '_RGB240.png')
        if destination.exists():
            print(f'SKIP existing {destination}', flush=True)
            continue
        run_one(image, destination, args.url.rstrip('/'), workflow, args.timeout)
