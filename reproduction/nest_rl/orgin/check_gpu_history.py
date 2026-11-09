"""Compare the CUDA learner history and encoder gradients against pinned NestRL."""

import os

import torch

from overcooked_setup import ROOT, activate_upstream


def main():
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    activate_upstream()
    torch.set_num_threads(1)

    from learning.storage_ import PeriodicHistoryStorage
    from cuda_history import CudaHistory, encode_cuda_history

    device = "cuda"
    lanes, rollout, horizon, period = 4, 24, 40, 5
    obs_size, actions = 73, 6
    model = torch.load(
        ROOT / "runs/rtx5070-l1-benchmark/ppo/latest.pt", map_location=device, weights_only=False,
    )
    encoder = model.encoder.eval().requires_grad_(True)
    reference = PeriodicHistoryStorage(
        num_processes=lanes, num_policies=1, history_storage_size=period, clear_period=period,
        refresh_interval=1, sample_size=None, has_rew_done=False, max_samples_per_period=None,
        step_mode=False, use_episodes=True, has_meta_time_step=False, include_current_episode=True,
        obs_shape=(obs_size,), act_shape=(), max_episode_length=horizon,
        merge_encoder_computation=True, last_episode_only=False, pop_oldest_episode=False,
    )
    reference.to(device)
    history = CudaHistory(lanes, rollout, (obs_size,), horizon, period, device)
    rng = torch.Generator().manual_seed(309)
    rows = torch.arange(lanes)
    lengths = torch.tensor([1, 3, 11, horizon])
    parameters = tuple(encoder.parameters())
    max_latent_error = max_grad_error = 0.0

    def check_sizes():
        for actual, expected in zip(history.history_sizes, reference.history_sizes, strict=True):
            torch.testing.assert_close(actual, torch.stack(expected), rtol=0, atol=0)

    for round_index in range(12):
        old_indices, new_indices = [], []
        for _ in range(rollout):
            old_indices.append(reference.get_all_current_indices())
            new_indices.append(history.get_all_current_indices())
            obs = torch.randn((lanes, obs_size), generator=rng)
            terminal = torch.randn((lanes, obs_size), generator=rng)
            action = torch.randint(actions, (lanes,), generator=rng)
            done = old_indices[-1][-1] + 1 == lengths
            finished = done.nonzero().flatten().numpy()
            ends_period = history.append(obs.to(device), action.to(device), terminal.to(device), done.to(device), finished)
            expected = torch.zeros(lanes, dtype=torch.bool)
            for lane in range(lanes):
                reference.add(lane, obs[lane], action[lane])
                if done[lane]:
                    reference.add(lane, terminal[lane], None)
                    expected[lane] = reference.finish_episode(lane)
            torch.testing.assert_close(ends_period.cpu(), expected, rtol=0, atol=0)
            assert list(history.current_episode) == reference.current_episode

        sample = torch.cat((torch.randperm(rollout * lanes, generator=rng), torch.arange(12)))
        processes = rows.repeat(rollout)[sample]
        old = tuple(torch.cat(parts)[sample] for parts in zip(*old_indices, strict=True))
        new = tuple(torch.cat(parts)[sample.to(device)] for parts in zip(*new_indices, strict=True))
        expected, _ = encoder((reference, (processes, *old)), processes)
        actual = encode_cuda_history(encoder, (history, (processes, *new)))
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
        weights = torch.randn(expected.shape, generator=rng).to(device)
        old_grad = torch.autograd.grad((expected * weights).mean(), parameters)
        new_grad = torch.autograd.grad((actual * weights).mean(), parameters)
        for original, batched in zip(old_grad, new_grad, strict=True):
            torch.testing.assert_close(batched, original, atol=1e-6, rtol=5e-4)
            max_grad_error = max(max_grad_error, (batched - original).abs().max().item())
        max_latent_error = max(max_latent_error, (actual - expected).abs().max().item())
        check_sizes()
        reference.trim()
        history.trim()
        check_sizes()
    print(f"Matched {12 * len(sample)} CUDA latents and 12 gradient batches across ring wraps. "
          f"Max latent error {max_latent_error:.3g}; max gradient error {max_grad_error:.3g}")


if __name__ == "__main__":
    main()
