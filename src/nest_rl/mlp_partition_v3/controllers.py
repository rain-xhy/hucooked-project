"""Select the recorded rule-controller semantics when replaying an experiment."""


def make_partners(env, roles, styles, version, learner_mask=None):
    if version == "serial-v1":
        from .legacy_partners import LegacyPartners
        return LegacyPartners(env, roles, styles)
    if version == "parallel-v2":
        from .partners import Partners
        return Partners(env, roles, styles, learner_mask=learner_mask)
    if version == "parallel-gpu-v3":
        from .gpu_partners import GpuPartners
        return GpuPartners(env, roles, styles, learner_mask=learner_mask)
    raise ValueError(f"Unknown controller version: {version}")
