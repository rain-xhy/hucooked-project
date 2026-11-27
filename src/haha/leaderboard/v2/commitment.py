"""An opt-in SharedKitchen scenario with consecutive joint work and chop ownership."""

from dataclasses import replace

from overcooked_ai_py.cooperative.shared import SharedKitchen
from overcooked_ai_py.cooperative.shared.kitchen import DIRECTIONS
from overcooked_ai_py.cooperative.shared.layouts import FLOOR


class CommitmentKitchen(SharedKitchen):
    joint_duration = 3
    rules = "consecutive-unload-and-chop-v1"

    def reset(self, seed=None):
        super().reset(seed)
        permutation = self.rng.permutation([order.kind for order in self.state.orders])
        for order, kind in zip(self.state.orders, permutation):
            order.kind = str(kind)
        self.state.metrics.update(joint_run=0, joint_pair=[], joint_run_resets=0, chopping_resets=0)
        self.validate_state()
        return self.observations(), self._info([], 0.0)

    def _requests(self, old, actions, state):
        requests = super()._requests(old, actions, state)
        run = old.metrics["joint_run"]
        past_pair = tuple(old.metrics["joint_pair"])
        pair = next((actors for name, actors, _, _ in requests if name == "unload"
                     and old.orders[old.cooker.item.order_id].kind == "joint"), ())
        current = 0
        if pair:
            current = 1 + (run if pair == past_pair else 0)
        if run and pair != past_pair:
            state.metrics["joint_run_resets"] += 1
        if pair and current < self.joint_duration:
            state.metrics["joint_run"] = current
            state.metrics["joint_pair"] = list(pair)
            requests = [request for request in requests if request[0] != "unload"]
        else:
            state.metrics["joint_run"] = 0
            state.metrics["joint_pair"] = []

        cutting = old.cutting
        if cutting.item is not None and cutting.item.kind == "raw_side" and cutting.progress:
            owner = cutting.item.prepared_by
            requests = [request for request in requests if request[0] != "chop" or request[1] == owner]
            if not any(request[0] == "chop" for request in requests):
                state.cutting.progress = 0
                state.cutting.item = replace(cutting.item, prepared_by=())
                state.metrics["chopping_resets"] += 1
        return requests

    def snapshot(self):
        return super().snapshot() | {"rules": self.rules}

    def restore(self, snapshot):
        if snapshot.get("rules") != self.rules:
            raise ValueError("Snapshot belongs to different SharedKitchen rules")
        metrics = snapshot.get("state", {}).get("metrics", {})
        required = {"joint_run", "joint_pair", "joint_run_resets", "chopping_resets"}
        if not required.issubset(metrics):
            raise ValueError("Snapshot does not contain the commitment state")
        super().restore(snapshot)

    def validate_state(self, state=None):
        state = self.state if state is None else state
        super().validate_state(state)
        # The parent validates the fresh reset before this variant initializes
        # the extra metrics. Later checkpoints carry the full commitment state.
        if "joint_run" not in state.metrics:
            return True
        run = state.metrics["joint_run"]
        pair = state.metrics["joint_pair"]
        if type(run) is not int or not 0 <= run < self.joint_duration:
            raise ValueError("Invalid consecutive unload step count")
        if (run and (len(pair) != 2 or len(set(pair)) != 2 or
                     any(i not in self.active for i in pair))) or (not run and pair):
            raise ValueError("Invalid consecutive unload pair")
        if state.cutting.item is not None and state.cutting.progress and len(state.cutting.item.prepared_by) != 1:
            raise ValueError("A cutting run belongs to one chef")
        return True


class ScatteredKitchen(CommitmentKitchen):
    rules = "consecutive-unload-and-chop-scattered-v1"

    def reset(self, seed=None):
        super().reset(seed)
        cells = sorted(FLOOR)
        locations = self.rng.choice(len(cells), size=len(self.active), replace=False)
        for i, location in zip(self.active, locations):
            self.state.chefs[i].position = cells[int(location)]
            self.state.chefs[i].orientation = DIRECTIONS[int(self.rng.integers(len(DIRECTIONS)))]
        self.validate_state()
        return self.observations(), self._info([], 0.0)
