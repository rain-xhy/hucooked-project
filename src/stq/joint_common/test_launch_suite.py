import json

from joint_common.launch_suite import run_job


def test_step_budget_only_job_has_no_process_timeout(tmp_path, monkeypatch):
    output = tmp_path / 'experiment'
    output.mkdir()
    (output / 'status.json').write_text(json.dumps({'state': 'completed'}))
    observed = {}

    class Process:
        pid = 123456

        def __init__(self, command, **kwargs):
            observed['command'] = command

        def wait(self, timeout=None):
            observed['timeout'] = timeout
            return 0

    monkeypatch.setattr('joint_common.launch_suite.subprocess.Popen', Process)
    run_job(tmp_path, {
        'name': 'unlimited', 'output': 'experiment', 'physical_gpu': 0,
        'module': 'joint_ppo.train', 'mode': 'rl', 'steps': 20_000_000,
        'memory_fraction': .008, 'max_seconds': 0,
    }, deadline=None)
    command = observed['command']
    assert command[command.index('--max-seconds') + 1] == '0'
    assert observed['timeout'] is None
    assert json.loads((output / 'queue.json').read_text())['state'] == 'exited'
