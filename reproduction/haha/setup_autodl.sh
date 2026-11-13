#!/usr/bin/env bash
# Run explicitly only after the other server task has finished. Installs, never trains.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
storage="${HAHA_STORAGE_ROOT:-/root/autodl-tmp/haha}"
case "$storage" in
  /root/autodl-tmp/*) ;;
  *) printf '%s\n' 'HAHA_STORAGE_ROOT must be a new task directory inside /root/autodl-tmp.' >&2; exit 1 ;;
esac
if [[ "$storage" == *'/../'* || "$storage" == */.. ]]; then
  printf '%s\n' 'Parent traversal is not supported in HAHA_STORAGE_ROOT.' >&2
  exit 1
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  printf '%s\n' 'This installer is for the AutoDL Linux x86_64 server only.' >&2
  exit 1
fi
if [[ ! -d /root/autodl-tmp ]]; then
  printf '%s\n' 'Expected AutoDL data disk /root/autodl-tmp is absent.' >&2
  exit 1
fi
free_kb="$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')"
if (( free_kb < 15 * 1024 * 1024 )); then
  printf '%s\n' 'Less than 15 GiB free on data disk. Inspect disk usage before installing; nothing is removed.' >&2
  exit 1
fi
conda_exe="$(command -v conda || true)"
if [[ -z "$conda_exe" && -x /root/miniconda3/bin/conda ]]; then
  conda_exe=/root/miniconda3/bin/conda
fi
if [[ -z "$conda_exe" ]]; then
  printf '%s\n' 'Conda was not found. Select its existing executable; this script does not install system software.' >&2
  exit 1
fi
env_prefix="$storage/env"
archive="$script_dir/cache/overcooked_ai-2cf0bd18d2523deb964b40ef99ef6bada19c8923.tar.gz"
if [[ ! -f "$archive" ]]; then
  printf 'Missing locally prepared author environment archive: %s\n' "$archive" >&2
  exit 1
fi
if [[ -e "$env_prefix" && ! -f "$env_prefix/.haha-reproduction-env" ]]; then
  printf 'Refusing to change an existing unmarked environment: %s\n' "$env_prefix" >&2
  exit 1
fi
mkdir -p "$storage/conda-pkgs" "$storage/pip-cache" "$storage/tmp"
export CONDA_PKGS_DIRS="$storage/conda-pkgs"
export PIP_CACHE_DIR="$storage/pip-cache"
export TMPDIR="$storage/tmp"
export PYTHONNOUSERSITE=1
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL="${HAHA_PIP_INDEX_URL:-https://pypi.org/simple}"
export PIP_EXTRA_INDEX_URL=
unset PYTHONPATH PYTHONHOME
if [[ ! -e "$env_prefix" ]]; then
  "$conda_exe" create --yes --prefix "$env_prefix" python=3.10 pip
  printf '%s\n' 'Isolated HAHA reproduction environment' > "$env_prefix/.haha-reproduction-env"
fi
python_exe="$env_prefix/bin/python"
"$python_exe" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
"$python_exe" -m pip install pip==24.0 setuptools==70.0.0 wheel==0.43.0
"$python_exe" -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# Use the bundled pinned source instead of fetching GitHub while GPU time is billed.
sed '/^overcooked-ai @ /d' "$script_dir/requirements.txt" > "$storage/requirements-runtime.txt"
"$python_exe" -m pip install -r "$storage/requirements-runtime.txt"
"$python_exe" -m pip install --no-build-isolation "$archive"
"$python_exe" -m pip check
printf '\nEnvironment ready: %s\n' "$env_prefix"
printf 'Project root: %s\n' "$(cd -- "$script_dir/../.." && pwd)"
printf '%s\n' 'No preprocessing, project validation or training has been started.'
