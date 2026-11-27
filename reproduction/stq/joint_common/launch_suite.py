"""Start independent experiments together on their explicitly assigned GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .runtime import atomic_json


def run_job(root, job, deadline):
    output = root / job['output']
    gpu = int(job['physical_gpu'])
    if gpu not in (0, 1, 2, 3):
        raise ValueError('Unexpected physical GPU assignment')
    if job['module'] != 'joint_ppo.train' and gpu != 1:
        raise ValueError('Paper reproductions are restricted to physical GPU 1')
    configured_seconds = float(job.get('max_seconds', 0))
    limits = [configured_seconds] if configured_seconds > 0 else []
    if deadline is not None:
        remaining = deadline - time.monotonic() - 120
        if remaining <= 0:
            raise RuntimeError('No time budget remains for ' + job['name'])
        limits.append(remaining)
    seconds = min(limits) if limits else None
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1', PYGAME_HIDE_SUPPORT_PROMPT='1',
                       SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy')
    command = [sys.executable, '-u', '-m', job['module'],
               '--output', str(output), '--mode', job['mode'],
               '--steps', str(job['steps']), '--physical-gpu', str(gpu),
               '--memory-fraction', str(job['memory_fraction']),
               '--max-seconds', str(int(seconds)) if seconds is not None else '0',
               *job.get('arguments', [])]
    started = datetime.now(timezone.utc).isoformat()
    with (output / 'train.log').open('a', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        (output / 'train.pid').write_text(str(process.pid), encoding='utf-8')
        atomic_json(output / 'queue.json', {
            'state': 'running', 'name': job['name'], 'pid': process.pid,
            'physical_gpu': gpu, 'execution': 'parallel',
            'started_utc': started, 'command': command,
        })
        print(json.dumps({'started': job['name'], 'pid': process.pid,
                          'physical_gpu': gpu}), flush=True)
        try:
            returncode = process.wait(timeout=seconds + 3600 if seconds is not None else None)
        except subprocess.TimeoutExpired:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError('Trainer exceeded its update-boundary time allowance: ' + job['name'])
    status_file = output / 'status.json'
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    atomic_json(output / 'queue.json', {
        'state': 'exited', 'returncode': returncode,
        'trainer_state': status.get('state'), 'started_utc': started,
        'ended_utc': datetime.now(timezone.utc).isoformat(), 'physical_gpu': gpu,
    })
    if returncode != 0 or not status or status.get('state') == 'failed':
        raise RuntimeError('Trainer failed: ' + job['name'])
    print(json.dumps({'finished': job['name'], 'returncode': returncode}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--only', nargs='+', help='Start these experiment names from the plan')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    root = Path(__file__).resolve().parents[1]
    jobs = plan['jobs']
    if args.only:
        jobs = [job for job in jobs if job['name'] in args.only]
    for job in jobs:
        (root / job['output']).mkdir(parents=True, exist_ok=True)
    max_hours = float(plan.get('max_hours', 0))
    deadline = time.monotonic() + max_hours * 3600 if max_hours > 0 else None
    failures = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(run_job, root, job, deadline): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append(job['name'])
                print(json.dumps({'failed': job['name'], 'error': str(error)}), flush=True)
    if failures:
        raise RuntimeError('Failed experiments: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
