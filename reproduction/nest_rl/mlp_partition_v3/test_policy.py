import unittest

import torch

from overcooked_ai_py.cooperative.partition import PartitionConfig
from overcooked_ai_py.cooperative.partition.model import CUT_TOMATO, EAST, TOMATO, WALL, cell

from . import OBS_DIM, V3Policy, V3PolicyKitchen, partner_action_ids
from .features import GROUPS


class PolicyTest(unittest.TestCase):
    def test_observation_semantics(self):
        env = V3PolicyKitchen(PartitionConfig(layout="counter"), device="cpu")
        state = env.state
        state["pos"][0] = torch.tensor([cell(5, 2), cell(8, 1), cell(9, 1)])
        state["facing"][0, 0] = EAST
        state["held"][0, 0] = TOMATO
        state["counter"][0, 1] = CUT_TOMATO
        obs = env.observations()
        groups = dict(zip((n for n, _ in GROUPS), obs.split([k for _, k in GROUPS], -1)))
        nearest = groups["movable_items"][0, 0].reshape(9, 3)
        torch.testing.assert_close(nearest[TOMATO-1], torch.tensor([0., 0., 1.]))
        torch.testing.assert_close(nearest[CUT_TOMATO-1], torch.tensor([1/13, 0., 1.]))
        torch.testing.assert_close(nearest[-1], torch.zeros(3))
        torch.testing.assert_close(groups["chopped_here"][0, 0], torch.tensor([0., 0., 1., 0.]))
        torch.testing.assert_close(partner_action_ids(torch.tensor([[1, 2, 3]])), torch.tensor([[15, 9, 8]]))

    def test_masks_and_gradients(self):
        torch.manual_seed(7)
        model = V3Policy()
        now = torch.rand(2, OBS_DIM)
        hist = torch.rand(2, 3, 4, OBS_DIM)
        ids = torch.randint(36, (2, 3, 4))
        pad = torch.zeros_like(ids, dtype=torch.bool)
        pad[:, 0, 2:] = True
        pad[:, 1] = True
        excluded = torch.tensor([[False, False, True]]).expand(2, -1)
        logp, val, cls = model(now, hist, ids, excluded, pad)
        torch.testing.assert_close(logp.exp().sum(-1), torch.ones(2))
        altered = hist.masked_fill((pad | excluded[..., None])[..., None], -99)
        changed_ids = ids.masked_fill(pad | excluded[..., None], -1)
        torch.testing.assert_close(model(now, altered, changed_ids, excluded, pad), (logp, val, cls))
        (-logp[:, 0].mean() + val.square().mean() + torch.nn.functional.cross_entropy(cls, torch.tensor([0, 2]))).backward()
        for block in (model.episode, model.period, model.actor, model.critic, model.partner_head):
            self.assertTrue(any(p.grad.abs().sum() > 0 for p in block.parameters()))
        outputs = model(now, hist, ids, excluded, torch.ones_like(pad))
        self.assertTrue(all(x.isfinite().all() for x in outputs))

    def test_five_episodes_and_forty_valid_steps(self):
        model = V3Policy(num_partner_classes=7)
        obs = torch.randn(1, OBS_DIM)
        history = torch.randn(1, 7, 50, OBS_DIM)
        ids = torch.randint(36, (1, 7, 50))
        pad = torch.zeros_like(ids, dtype=torch.bool)
        pad[:, :, 46:] = True
        excluded = pad.all(-1)
        excluded[:, -1] = True
        out = model(obs, history, ids, excluded, pad)
        expected = model(obs, history[:, 1:6, 6:46], ids[:, 1:6, 6:46],
                         excluded[:, 1:6], pad[:, 1:6, 6:46])
        torch.testing.assert_close(out, expected)
        self.assertEqual(out[2].shape, (1, 7))
        torch.testing.assert_close(out[2], model(obs + 1, history, ids, excluded, pad)[2])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_real_rollouts(self):
        torch.manual_seed(5)
        for layout in ("sync", "counter"):
            cfg = PartitionConfig(layout=layout, horizon=12)
            cpu = V3PolicyKitchen(cfg, device="cpu", num_envs=4)
            gpu = V3PolicyKitchen(cfg, num_envs=4)
            policy = V3Policy().cuda().eval()
            obs, _ = gpu.reset(19)
            cpu_obs, _ = cpu.reset(19)
            hist = obs.new_zeros(12, 1, 12, OBS_DIM)
            ids = torch.full((12, 1, 12), -1, device="cuda", dtype=torch.long)
            pad = ids == -1
            with torch.no_grad():
                for t in range(12):
                    torch.testing.assert_close(obs.cpu(), cpu_obs)
                    current = obs.flatten(0, 1)
                    logp, val, _ = policy(current, hist, ids, pad.all(-1), pad)
                    self.assertEqual(val.shape, (12, 1))
                    act = logp.exp().multinomial(1).reshape(4, 3)
                    hist[:, 0, t] = current
                    ids[:, 0, t] = partner_action_ids(act).flatten()
                    pad[:, 0, t] = False
                    obs, *_, done, _ = gpu.step(act)
                    cpu_obs, *_ = cpu.step(act.cpu())
                self.assertTrue(done.all())
                torch.testing.assert_close(obs.cpu(), cpu_obs)
