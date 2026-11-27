"""Roll out the original Overcooked MDP with sparse scoring."""

from overcooked_ai_py.mdp.actions import Action


class Episode:
    def __init__(self, mdp, state, horizon):
        self.mdp, self.state, self.horizon = mdp, state, horizon

    def step(self, actions):
        joint_action = [Action.INDEX_TO_ACTION[int(action)] for action in actions]
        self.state, info = self.mdp.get_state_transition(self.state, joint_action)
        reward = sum(info["sparse_reward_by_agent"])
        done = self.state.timestep >= self.horizon or self.mdp.is_terminal(self.state)
        return reward, done
