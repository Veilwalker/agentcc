"""Read the supplied image archive and test its Python without Docker/chroot.

Only Python, headers and shared libraries are copied into a new project folder.
No package manager, system installation or external directory is touched.
"""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / 'materials/runtime-1.1.1'


def selected(name):
    return (name.startswith(('usr/local/lib/python3.11/',
                             'usr/local/include/python3.11/',
                             'usr/lib/x86_64-linux-gnu/'))
            or name == 'usr/local/bin/python3.11'
            or name.startswith('usr/local/lib/libpython3.11.so'))


def extract(archive, root):
    root = root.resolve()
    if not root.is_relative_to(PROJECT / 'materials') or root.exists():
        raise ValueError('runtime output must be a new directory under materials')
    root.mkdir(parents=True)
    links, count, whiteouts = {}, 0, []
    with tarfile.open(archive) as outer:
        manifest = json.load(outer.extractfile('manifest.json'))[0]
        config = json.load(outer.extractfile(manifest['Config']))
        if config['architecture'] != 'amd64' or config['os'] != 'linux':
            raise ValueError('this runtime check expects Linux/AMD64')
        for layer in manifest['Layers']:
            with tarfile.open(fileobj=outer.extractfile(layer), mode='r|*') as stream:
                for member in stream:
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or '..' in path.parts:
                        raise ValueError('unsafe image member')
                    name = str(path)
                    if not selected(name) or '__pycache__' in path.parts:
                        continue
                    if path.name.startswith('.wh.'):
                        # Image layer deletions apply only to this fresh extraction.
                        # Symlinks are deferred, so no archive link is followed.
                        victim = path.parent if path.name == '.wh..wh..opq' else path.with_name(path.name[4:])
                        target = root / str(victim)
                        if not target.resolve().is_relative_to(root):
                            raise ValueError('unsafe image deletion')
                        if target.is_dir():
                            shutil.rmtree(target)
                        elif target.exists():
                            target.unlink()
                        prefix = str(victim)
                        links = {k: v for k, v in links.items() if k != prefix and not k.startswith(prefix + '/')}
                        whiteouts.append(str(victim))
                        continue
                    target = root / name
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with stream.extractfile(member) as source, target.open('wb') as destination:
                            shutil.copyfileobj(source, destination)
                        target.chmod(0o755 if member.mode & 0o111 else 0o644)
                        links.pop(name, None)
                        count += 1
                    elif member.issym() or member.islnk():
                        link = PurePosixPath(member.linkname)
                        relative = (root / str(link).lstrip('/') if link.is_absolute()
                                    or member.islnk() else target.parent / str(link))
                        resolved = relative.resolve()
                        if not resolved.is_relative_to(root):
                            raise ValueError('image link escapes runtime directory')
                        links[name] = resolved
        for name, resolved in links.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(os.path.relpath(resolved, target.parent))
        evidence = {'image_tags': manifest['RepoTags'], 'config_digest': manifest['Config'],
                    'architecture': config['architecture'], 'files_copied': count,
                    'layer_deletions': whiteouts,
                    'scope': 'Selected runtime files only; no container isolation'}
        (root / 'extraction.json').write_text(json.dumps(evidence, indent=2) + '\n')
        print(json.dumps(evidence))


def interpreter_command(root=None):
    """Keep subprocess-based tests on the same extracted runtime, if selected."""
    root = root or os.environ.get('AGENTCC_TEST_RUNTIME_ROOT')
    if root is None:
        return [sys.executable]
    root = Path(root).resolve()
    if not root.is_relative_to(PROJECT / 'materials'):
        raise ValueError('runtime must stay under project materials')
    library = root / 'usr/lib/x86_64-linux-gnu'
    interpreter = root / 'usr/local/bin/python3.11'
    return [str(library / 'ld-linux-x86-64.so.2'), '--library-path',
            str(root / 'usr/local/lib') + ':' + str(library), str(interpreter)]


def run(root, arguments):
    root = root.resolve()
    command = interpreter_command(root) + arguments
    environment = dict(os.environ, PYTHONHOME=str(root / 'usr/local'),
                       PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(PROJECT),
                       AGENTCC_TEST_RUNTIME_ROOT=str(root))
    return subprocess.run(command, cwd=PROJECT, env=environment).returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--extract', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if options.extract:
        extract(options.extract, options.root)
    if options.arguments:
        args = options.arguments[1:] if options.arguments[0] == '--' else options.arguments
        raise SystemExit(run(options.root, args))
