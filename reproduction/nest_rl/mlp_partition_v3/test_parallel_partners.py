import unittest
import torch
from overcooked_ai_py.cooperative.partition import PartitionConfig
from .kitchen import V3PolicyKitchen
from .partners import Partners

class ParallelPartnersTest(unittest.TestCase):
    def test_two_distinct_jobs_and_learner_mask(self):
        env = V3PolicyKitchen(PartitionConfig(layout='counter', horizon=400), num_envs=3, device='cpu')
        env.reset(seed=37)
        p = Partners(env, torch.arange(3), torch.arange(3), learner_mask=torch.tensor([True, False, True]))
        a = p.actions()
        self.assertEqual(p.version, 'parallel-v2')
        self.assertTrue((p.job_ticket_ids[:, 0] != p.job_ticket_ids[:, 1]).all())
        self.assertEqual(a[0, p.learner_ids[0]].item(), 4)
        self.assertEqual(a[2, p.learner_ids[2]].item(), 4)

    def make(self, n=3, device='cpu'):
        env = V3PolicyKitchen(PartitionConfig(layout='counter', horizon=400), num_envs=n, device=device)
        env.reset(seed=37)
        p = Partners(env, torch.arange(n, device=device) % 3, torch.arange(n, device=device) % 3)
        return env, p

    def test_seeded_styles_deliver_each_recipe_despite_missed_actions(self):
        env, p = self.make(9)
        target = torch.arange(9) // 3
        env.state['recipe'][:] = target[:, None]
        env.state['due'][:] = 300
        delivered = torch.zeros(9, dtype=torch.bool)
        right_board_used = False
        for step in range(400):
            actions = p.actions()
            if step % 11 == 3:
                actions[actions == 5] = 4
            _, _, _, _, info = env.step(actions)
            delivered |= (info['delivered'] & (info['recipe'] == target[:, None])).any(1)
            right_board_used |= bool((env.state['board'][:, 2:] != 0).any())
        self.assertTrue(delivered.all(), delivered)
        self.assertTrue(right_board_used)

    def test_two_pots_cook_and_cross_side_deliver(self):
        env, p = self.make()
        env.state['recipe'][:] = torch.tensor([0, 1, 2])
        env.state['due'][:] = 300
        # Deliberately cook each meal opposite its delivery side.
        env.state['mix'][:] = torch.tensor([5, 2])
        heated_together = False
        delivered = torch.zeros((3, 2), dtype=torch.bool)
        for _ in range(100):
            before = env.state['heat'].clone()
            _, _, _, _, info = env.step(p.actions())
            heated_together |= bool((env.state['heat'] > before).all(1).any())
            for r in (0, 1):
                delivered[:, r] |= (info['delivered'] & (info['recipe'] == r)).any(1)
        self.assertTrue(heated_together)
        self.assertTrue(delivered.all(), delivered)

    def test_checkpoint_observed_progress_and_full_counter(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            env, p = self.make(device=device)
            for _ in range(23):
                env.step(p.actions())
            saved = p.state_dict()
            expected = p.actions().clone()
            p.reset()
            p.load_state_dict(saved)
            torch.testing.assert_close(p.actions(), expected)
            # Suppressed interactions must not advance task state in a static kitchen.
            first = p.actions().clone()
            for _ in range(5):
                torch.testing.assert_close(p.actions(), first)
            env.state['counter'][:] = torch.tensor([1, 2, 3, 4, 7], device=device)
            env.state['ids'] += 100
            deliveries = 0
            for _ in range(100):
                actions = p.actions()
                for lane, records in enumerate(p.reservations):
                    targets = [r[1] for r in records if r[1] != 0]
                    self.assertEqual(len(targets), len(set(targets)))
                _, _, _, _, info = env.step(actions)
                deliveries += int(info['delivered'].sum())
            self.assertGreater(deliveries, 0)
            self.assertTrue(torch.isfinite(actions).all())

    def test_expired_partial_job_does_not_fetch_new_ingredients(self):
        env, p = self.make(1)
        env.state['recipe'][:] = 0
        env.state['mix'][0, 0] = 8
        p.actions()
        self.assertEqual(p.job_ticket_ids[0, 0], -1)
        # No beef recipe ticket exists; its orphan pot may use existing food,
        # but must not cause a new potato source reservation.
        from overcooked_ai_py.cooperative.partition.model import POTATO, SOURCES
        self.assertNotIn(SOURCES[POTATO], p.destinations[0])


    def test_all_recipe_pairs_claim_both_pots(self):
        env, p = self.make(6)
        pairs = torch.tensor([[0,0,0],[1,1,1],[2,2,2],[0,1,0],[0,2,0],[1,2,1]])
        env.state['recipe'][:] = pairs
        # FIFO ranking keeps the first two slots; mapping may exchange pots.
        p.styles[:] = 0
        p.actions()
        for lane in range(6):
            self.assertEqual(sorted(p.job_ticket_ids[lane].tolist()), [0,1])
            self.assertEqual(sorted(p.job_recipes[lane].tolist()), sorted(pairs[lane,:2].tolist()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cpu_cuda_actions_and_states_match(self):
        cpu, a = self.make(6, 'cpu')
        gpu, b = self.make(6, 'cuda')
        for _ in range(110):
            x, y = a.actions(), b.actions()
            torch.testing.assert_close(x, y.cpu())
            cpu.step(x)
            gpu.step(y)
            for key in cpu.state:
                torch.testing.assert_close(cpu.state[key], gpu.state[key].cpu())

    def test_shared_source_and_idle_diagnostics(self):
        from overcooked_ai_py.cooperative.partition.model import EGG, SOURCES
        env, p = self.make(3)
        env.state['recipe'][:] = 0
        actions = p.actions()
        for lane in range(3):
            self.assertLessEqual(int((p.destinations[lane] == SOURCES[EGG]).sum()), 1)
        self.assertEqual(p.idle_actionable.shape, (3, 3))
        self.assertTrue((~p.idle_actionable | (actions.numpy() == 4)).all())
        # Reserving a learner's held egg prevents another source fetch for
        # that same ingredient, even though the planner cannot command them.
        p.learner_mask[:] = True
        p.roles[:] = 1
        p._refresh_learner_ids()
        env.state['held'][torch.arange(3), p.learner_ids] = EGG
        env.state['mix'][:, 0] = 5
        p.actions()
        for lane in range(3):
            self.assertNotIn(SOURCES[EGG], p.destinations[lane])
            self.assertFalse(p.idle_actionable[lane, p.learner_ids[lane]])
