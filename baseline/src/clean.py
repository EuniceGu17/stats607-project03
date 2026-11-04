"""Remove generated stage directories inside this project's results tree only."""

from pathlib import Path
import shutil

from .common import ROOT, arguments


def clean(output):
    output = Path(output).resolve()
    results = (ROOT / "results").resolve()
    if output == results or not output.is_relative_to(results):
        raise ValueError("Clean accepts only a profile directory below this project's results/")
    for name in ["raw", "summaries", "figures"]:
        target = output / name
        if target.is_symlink() or not target.resolve().is_relative_to(results):
            raise ValueError("Refusing to clean a symlink or a path outside results/")
        if target.exists():
            shutil.rmtree(target)


if __name__ == "__main__":
    clean(arguments(__doc__).output)
