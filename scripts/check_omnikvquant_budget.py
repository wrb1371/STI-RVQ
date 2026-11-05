"""Audit measured input geometries without loading weights or using a GPU.

Input: JSON list of records containing id, video, model, dataset, duration_group,
layers, kv_heads, head_dim, video_tokens, and budget_bytes (or budget_mib).
Use actual processor geometry, including temporal and spatial merging.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ar50kv.budget import check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = json.loads(Path(args.geometry_manifest).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('The geometry manifest must be a nonempty JSON list')
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(check(row), ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
