import unittest
import numpy as np
from .envs import KitchenBatch, KitchenSlot, potential
from overcooked_ai_py.cooperative.shared.controller import ScriptedController


class TaskTests(unittest.TestCase):
    def test_parallel_matches_native_schema(self):
        with KitchenBatch(2, seed=14) as a, KitchenBatch(2, seed=14, workers=1) as b:
            x,y=a.reset(),b.reset()
            self.assertEqual(x['vector'].shape,(2,3,866))
            self.assertEqual(x['state'].shape,(2,863))
            self.assertEqual(x['spatial'].shape,(2,3,26,7,7))
            rng=np.random.default_rng(5)
            for _ in range(10):
                for key in x:np.testing.assert_array_equal(x[key],y[key])
                actions=rng.integers(0,6,(2,3))
                x,r,d,_=a.step(actions);y,s,e,_=b.step(actions)
                np.testing.assert_array_equal(r,s);np.testing.assert_array_equal(d,e)

    def test_metrics_and_raw_reward_are_native(self):
        slot=KitchenSlot(14,0,'standard',False,.9985);slot.reset()
        controller=ScriptedController(slot.core)
        for _ in range(300):
            _,reward,done,info=slot.step(controller.actions(slot.core))
            self.assertEqual(reward,info['team_reward'])
            self.assertEqual(info['training_shaping'],0.)
            if done:break
        summary=info['episode_summary']
        for key,value in info['metrics'].items():self.assertEqual(summary[key],value)
        self.assertGreater(summary['team_score'],0)
        self.assertGreater(summary['joint_successes'],0)

    def test_shaping_telescopes_at_terminal(self):
        gamma=.9985
        slot=KitchenSlot(1,0,'standard',True,gamma);slot.reset()
        start=potential(slot.core)
        controller=ScriptedController(slot.core)
        discounted=0.
        for i in range(300):
            _,_,done,info=slot.step(controller.actions(slot.core))
            discounted+=gamma**i*info['training_shaping']
            if done:break
        self.assertAlmostEqual(discounted,-.2*start,places=5)


if __name__ == '__main__':
    unittest.main()
