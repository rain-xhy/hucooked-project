from overcooked_ai_py.cooperative.partition import PartitionConfig, TorchPartitionKitchen

from .features import V3Features


class V3PolicyKitchen(TorchPartitionKitchen):
    """Drop-in batched v3 environment emitting self-centric policy observations."""

    def __init__(self, config=None, *, device="cuda", num_envs=1):
        config = PartitionConfig() if config is None else config
        self.features = V3Features(config).to(device)
        super().__init__(config, device=device, num_envs=num_envs)

    def observations(self):
        return self.features(self.state)
