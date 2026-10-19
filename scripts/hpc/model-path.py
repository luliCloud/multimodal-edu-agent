"""Resolve a local Hugging Face snapshot without loading Hugging Face or models."""

import sys
from pathlib import Path


def main() -> int:
    cache, repository = sys.argv[1:]
    model = Path(cache) / ("models--" + repository.replace("/", "--"))
    snapshots = model / "snapshots"
    reference = model / "refs" / "main"
    if reference.is_file():
        revision = reference.read_text().strip()
        if revision and Path(revision).name == revision:
            selected = snapshots / revision
            if selected.is_dir():
                print(selected.absolute())
                return 0
    choices = sorted(path for path in snapshots.glob("*") if path.is_dir())
    if len(choices) == 1:
        print(choices[0].absolute())
        return 0
    if len(choices) > 1:
        print(f"Multiple cached snapshots for {repository}; set its model path in .hpc.env.",
              file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
