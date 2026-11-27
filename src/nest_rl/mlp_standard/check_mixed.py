"""Verify each learner slot and delivery metrics against native MixedKitchen."""

from standard_setup import activate


def main():
    activate(legacy_gpu=True)

    import torch
    from overcooked_ai_py.cooperative.kitchen import MixedKitchen
    from mixed_kitchen import CONFIG, MixedVecEnv, ROLES

    torch.set_num_threads(1)
    vec = MixedVecEnv(3, "cuda", 1, player=1)
    references = [MixedKitchen(CONFIG) for _ in ROLES]
    for lane, (role, reference) in enumerate(zip(ROLES, references, strict=True)):
        vec.env_method("set_opponent", role, indices=lane)
        reference.reset(seed=1 + lane)
    obs = vec.reset()
    assert obs.shape == (3, 190)
    torch.testing.assert_close(obs[:, -3:], torch.eye(3, device="cuda"))

    complete = set()
    for _ in range(CONFIG.horizon):
        team = vec.rule_partner.act()
        learner = team[vec.rows, vec.roles]
        _, rewards, done, infos = vec.step(learner)
        for lane, reference in enumerate(references):
            if lane in complete:
                continue
            _, reward, terminal, truncated, _ = reference.step(tuple(team[lane].tolist()))
            assert abs(rewards[lane].item() - reward) < 1e-6
            assert done[lane] == (terminal or truncated)
            if done[lane]:
                latencies = reference.state.metrics["latencies"]
                assert infos[lane]["delivery_count"] == len(latencies)
                assert infos[lane]["delivery_steps"] == sum(latencies)
                assert infos[lane]["termination_info"].endswith("completed")
                complete.add(lane)
        if len(complete) == len(ROLES):
            break
    assert len(complete) == len(ROLES)
    print("All three learner slots matched native rewards, termination and order-to-delivery timing.")


if __name__ == "__main__":
    main()
