"""Materialize the pinned upstream sources needed by the leaderboard."""

import argparse
from pathlib import Path
import subprocess
import tarfile
from tempfile import TemporaryDirectory

from run import PROJECT, REVISIONS


REMOTE = {
    "original": "git@github.com:StephAO/overcooked_ai.git",
    "hucooked": "git@github.com:rain-xhy/hucooked.git",
}
COOPERATIVE = ("src/overcooked_ai_py/__init__.py", "src/overcooked_ai_py/cooperative")


def contains(repository, revision):
    return repository.is_dir() and subprocess.run(
        ["git", "-C", str(repository), "cat-file", "-e", f"{revision}^{{commit}}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def export(repository, revision, archive, version):
    files = () if version == "original" else (*COOPERATIVE, "LICENSE", *(("v3.md",) if version == "v3" else ()))
    archive.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(repository), "archive", "-o", str(archive), revision, *files], check=True)


def prepare(version, checkout):
    revision = REVISIONS[version]
    cache = PROJECT / ".environment"
    cache.mkdir(exist_ok=True)
    destination = cache / version
    if destination.exists():
        marker = destination / "revision"
        if not marker.is_file() or marker.read_text().strip() != revision:
            raise SystemExit(f"Unexpected revision in {destination}")
        if not (destination / "src/overcooked_ai_py/__init__.py").is_file():
            raise SystemExit(f"Missing environment source in {destination}")
        print(f"{version} {revision} -> {destination}")
        return
    with TemporaryDirectory(dir=cache) as work:
        build = Path(work)
        source = build / "src-tree"
        source.mkdir()
        bundle = build / "environment.tar"
        repository = checkout or (PROJECT / "hucooked")
        if not contains(repository, revision):
            if checkout:
                raise SystemExit(f"Commit {revision} is missing from {checkout}")
            repository = build / "upstream"
            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            subprocess.run(["git", "-C", str(repository), "fetch", "--depth", "1", "--no-tags",
                            REMOTE["original" if version == "original" else "hucooked"], revision], check=True)
        export(repository, revision, bundle, version)
        with tarfile.open(bundle) as source_archive:
            source_archive.extractall(source, filter="data")
        (source / "revision").write_text(revision + "\n")
        source.rename(destination)
    print(f"{version} {revision} -> {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=(*REVISIONS, "all"))
    parser.add_argument("--checkout", type=Path, help="Use an existing Git checkout containing the pinned revision")
    args = parser.parse_args()
    for version in REVISIONS if args.version == "all" else (args.version,):
        prepare(version, args.checkout)


if __name__ == "__main__":
    main()
