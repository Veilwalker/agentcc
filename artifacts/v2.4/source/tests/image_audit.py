"""Inspect exported images without Docker; extract only the application files.

Never follows archive links or executes image contents. Frozen exports remain
unchanged; the extracted copy lets other tests run the actual shipped binary.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

PROJECT = Path(__file__).resolve().parents[1]


def audit(version, output):
    archive = PROJECT.parent / 'agentcc-images' / f'agentcc-team-{version}.tar'
    root = (output / f'image-{version}').resolve()
    if not root.is_relative_to(PROJECT) or (root.exists() and any(root.iterdir())):
        raise ValueError('extraction requires a new project directory')
    root.mkdir(parents=True, exist_ok=True)
    with archive.open('rb') as source:
        hasher = hashlib.sha256()
        for block in iter(lambda: source.read(1024*1024), b''):
            hasher.update(block)
        digest = hasher.hexdigest()
    with tarfile.open(archive) as outer:
        entries = json.load(outer.extractfile('manifest.json'))
        if len(entries) != 1:
            raise ValueError('expected exactly one image')
        entry = entries[0]
        config = json.load(outer.extractfile(entry['Config']))
        for layer in entry['Layers']:
            with tarfile.open(fileobj=outer.extractfile(layer), mode='r|*') as stream:
                for member in stream:
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or '..' in path.parts:
                        raise ValueError('unsafe image path')
                    if not str(path).startswith('app/src/'):
                        continue
                    if path.name.startswith('.wh.') or member.issym() or member.islnk():
                        raise ValueError('application overlay/link requires explicit review')
                    if member.isfile():
                        target = root / path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(stream.extractfile(member).read())
        files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((root/'app/src').rglob('*')) if p.is_file()}
        snapshot = PROJECT / f'artifacts/v{version}/source/src'
        same = all((root/'app/src'/p.name).is_file() and
                   p.read_bytes() == (root/'app/src'/p.name).read_bytes()
                   for p in snapshot.glob('*.py'))
        extras = [p.name for p in (root/'app/src').glob('*.py') if not (snapshot/p.name).exists()]
        env = config['config'].get('Env', [])
        safe_env = [x for x in env if x.startswith('AGENTCC_CAPACITY_BACKEND=')]
        result = dict(version=version, archive=str(archive), sha256=digest,
                      image_id=entry['Config'], tags=entry['RepoTags'],
                      architecture=config['architecture'], os=config['os'],
                      user=config['config']['User'], command=config['config']['Cmd'],
                      backend_env=safe_env, python_matches_snapshot=same,
                      base_image_extra_python=sorted(extras), files=files)
        (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--versions',nargs='+',default=['1.6','1.9','1.10'])
    parser.add_argument('--output',default='artifacts/v1.11/analysis/images')
    args=parser.parse_args()
    for version in args.versions:
        audit(version,PROJECT/args.output)
