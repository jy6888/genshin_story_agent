"""Validate local-only assets before creating a live model or sending requests."""
from pathlib import Path


def require_local_data(root):
    from story_agent.texts_tl4 import KINDS
    root = Path(root).resolve()
    directories = ['.runtime/istaroth-checkpoint/text/agd_' + kind
                   for kind in ('book', 'quest', *KINDS)]
    directories.append('.runtime/book-map/v1')
    files = ['data/manifests/book-map-v1.json',
             '.runtime/similar-passages/v1/manifest.json',
             '.runtime/similar-passages/v1/neighbors.sqlite',
             '.runtime/vector-recall/v1/units.json',
             '.runtime/vector-recall/v1/books.json',
             '.runtime/loregraph-8cfd3746b135b87433c27c188b23810a544d0ee9/graphlink_genshin_agent/character_graph.json']
    missing = [name + '/' for name in directories if not (root/name).is_dir()]
    missing += [name for name in files if not (root/name).is_file()]
    missing += [name + '/ (empty text directory)' for name in directories
                if '/text/' in name and (root/name).is_dir() and not any((root/name).glob('*.txt'))]
    if missing:
        raise FileNotFoundError('Local research assets are missing or empty. No model request was sent. '
            'The source distribution excludes game text, maps, caches and community graph. '
            'Required paths relative to the repository root: ' + '; '.join(missing))
