"""Apply linear learning-rate warmup at PPO optimizer steps."""

from functools import wraps


def install_lr_warmup(ppo_class, warmup_steps):
    original_update = ppo_class.update
    completed_steps = 0
    target_rates = None
    handle = None

    def multiplier():
        if warmup_steps == 1:
            return 1.0
        return 0.1 + 0.9 * min(1.0, completed_steps / (warmup_steps - 1))

    @wraps(original_update)
    def update(agent, *args, **kwargs):
        nonlocal target_rates, handle
        if target_rates is None:
            target_rates = [group["lr"] for group in agent.optimizer.param_groups]

            def before_step(optimizer, step_args, step_kwargs):
                nonlocal completed_steps
                factor = multiplier()
                for group, target in zip(optimizer.param_groups, target_rates):
                    group["lr"] = target * factor
                completed_steps += 1

            handle = agent.optimizer.register_step_pre_hook(before_step)

        start_rate = target_rates[0] * multiplier()
        info = original_update(agent, *args, **kwargs)
        if handle is not None and completed_steps >= warmup_steps:
            handle.remove()
            handle = None
        info["learning_rate_start"] = start_rate
        info["learning_rate"] = agent.optimizer.param_groups[0]["lr"]
        return info

    ppo_class.update = update
