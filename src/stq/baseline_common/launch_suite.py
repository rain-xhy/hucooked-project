"""Launch training jobs concurrently on physical GPU 1."""
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


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(temp, path)


def run_job(root, job, env, deadline):
    out = root / job['output']
    seconds = min(float(job['max_seconds']), deadline - time.time() - 180)
    if seconds <= 0:
        raise RuntimeError('Time budget exhausted before starting ' + job['method'])
    command = [sys.executable, '-u', '-m', job['method'] + '.train',
               '--output', str(out), '--steps', str(job['steps']),
               '--max-seconds', str(int(seconds)), *job.get('arguments', [])]
    started = datetime.now(timezone.utc).isoformat()
    with (out / 'train.log').open('w', encoding='utf-8') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, cwd=root, start_new_session=True)
        (out / 'train.pid').write_text(str(child.pid))
        write_json(out / 'queue.json', {'state': 'running', 'pid': child.pid,
                                       'started_utc': started, 'command': command,
                                       'execution': 'parallel', 'physical_gpu': 1})
        print(json.dumps({'started': job['method'], 'pid': child.pid}), flush=True)
        try:
            result = child.wait(timeout=seconds + int(job.get('checkpoint_grace_seconds', 600)))
        except subprocess.TimeoutExpired:
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            write_json(out / 'queue.json', {'state': 'external_time_limit', 'started_utc': started})
            raise RuntimeError('Trainer exceeded its time budget: ' + job['method'])
    status_path = out / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    write_json(out / 'queue.json', {'state': 'exited', 'returncode': result,
                                   'trainer_state': status.get('state'),
                                   'started_utc': started,
                                   'ended_utc': datetime.now(timezone.utc).isoformat()})
    if result != 0 or not status or status.get('state') == 'failed':
        raise RuntimeError('Trainer failed: ' + job['method'])
    print(json.dumps({'finished': job['method'], 'returncode': result}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    if plan.get('physical_gpu', 1) != 1 or plan.get('execution', 'parallel') != 'parallel':
        raise ValueError('This suite runs concurrently on physical GPU 1')
    root = Path(__file__).resolve().parents[1]
    deadline = time.time() + float(plan.get('max_hours', 44)) * 3600
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', PYGAME_HIDE_SUPPORT_PROMPT='1',
               SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy')
    jobs = plan['jobs']
    outputs = [root / job['output'] for job in jobs]
    if not jobs or len(set(outputs)) != len(outputs):
        raise ValueError('Expected jobs with distinct output directories')
    for out in outputs:
        if any((out / name).exists() for name in (
                'config.json', 'initial.pt', 'latest.pt', 'final.pt', 'status.json', 'metrics.jsonl', 'train.pid')):
            raise FileExistsError('Refusing to overwrite experiment: ' + str(out))
    for job, out in zip(jobs, outputs):
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / 'queue.json', {'state': 'starting', 'method': job['method']})
    failures = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(run_job, root, job, env, deadline): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append(job['method'])
                print(json.dumps({'failed': job['method'], 'error': str(error)}), flush=True)
    if failures:
        raise RuntimeError('Failed jobs: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
