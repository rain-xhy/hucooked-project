"""Schema-1 visual projection with lossless V3 state attached to every snapshot.

Unity's original assets cannot distinguish potato/egg/beef or cutting boards.
The projection is explicitly approximate; native_state/native_info retain the
authoritative values so downstream V3 renderers need not infer any rules.
"""
from pathlib import Path
from overcooked_ai_py.cooperative.partition.model import (
    WIDTH, HEIGHT, FLOOR, BOARDS, POTS, WALL, SOURCES, SERVERS, RECIPE_NAMES, RECIPES,
)

ORIENTATIONS = ((0, -1), (0, 1), (1, 0), (-1, 0))
ITEM_NAMES = ("empty", "tomato", "potato", "egg", "beef", "cut_tomato", "cut_potato", *RECIPE_NAMES)


def ingredients(mix):
    return [name for bit, name in ((1, "tomato"), (2, "onion"), (4, "onion"), (8, "onion")) if mix & bit]


def xy(cell):
    return [int(cell % WIDTH), int(cell // WIDTH)]


def object_view(item, position):
    if not item:
        return None
    name = "soup" if item >= 7 else "tomato" if item in (1, 5) else "onion"
    obj = dict(name=name, position=position, native_item_id=item, native_item=ITEM_NAMES[item])
    if name == "soup":
        obj.update(_ingredients=[dict(name=n, position=position) for n in ingredients(RECIPES[item-7])], cooking_tick=4, cook_time=4, is_idle=False, is_ready=True, is_cooking=False)
    return obj


def snapshot(native):
    players = [dict(position=xy(p), orientation=list(ORIENTATIONS[f]), held_object=object_view(h, xy(p)))
               for p, f, h in zip(native["pos"], native["facing"], native["held"])]
    objects = [obj for positions, items in ((WALL, native["counter"]), (BOARDS, native["board"]))
               for pos, item in zip(positions, items) if (obj := object_view(item, xy(pos))) is not None]
    for j, pos in enumerate(POTS):
        if native["mix"][j]:
            heat = native["heat"][j]
            objects.append(dict(name="soup", position=xy(pos), _ingredients=[dict(name=n, position=xy(pos)) for n in ingredients(native["mix"][j])], cooking_tick=heat if heat else -1,
                                cook_time=4, is_idle=heat == 0, is_cooking=0 < heat < 4, is_ready=heat == 4,
                                native_mix=native["mix"][j], native_locked=native["locked"][j]))
    orders = [dict(ingredients=ingredients(RECIPES[r]), id=native["ids"][i], recipe=r,
                   issued=native["issued"][i], due=native["due"][i], side=1 if r==0 else 0)
              for i,r in enumerate(native["recipe"])]
    return dict(players=players, objects=objects, timestep=native["tick"][0], all_orders=orders, bonus_orders=[], native_state=native)


class EvaluationReplays:
    def __init__(self, env, metadata, roles, styles):
        from better_overcooked_replay.protocol import make_header
        self.env = env
        grid = [[" " if y*WIDTH+x in FLOOR else "X" for x in range(WIDTH)] for y in range(HEIGHT)]
        for pos in POTS:
            grid[pos // WIDTH][pos % WIDTH] = "P"
        for pos in SERVERS:
            grid[pos // WIDTH][pos % WIDTH] = "S"
        for item, pos in SOURCES.items():
            grid[pos // WIDTH][pos % WIDTH] = "T" if item == 1 else "O"
        terrain = ["".join(row) for row in grid]
        self.data = []
        native = {k: v[:9].cpu().tolist() for k, v in env.state.items()}
        for lane in range(9):
            chef_ids = sorted(range(3), key=lambda i: -1 if native["pos"][lane][i] % WIDTH < 6 else native["pos"][lane][i])
            header = make_header(terrain, "v3-counter-projection", dict(metadata, horizon=env.config.horizon,
                complete=True, dynamics="partition-v3", role=roles[lane//3], style=styles[lane%3],
                learner_id=chef_ids[lane//3], learner_role=roles[lane//3], learner_externally_controlled=metadata.get("mode") != "script",
                projection="Original Unity assets approximate potato, egg, beef as onion and boards as counters; native_state is authoritative.",
                native_item_names=list(ITEM_NAMES), native_boards=[xy(p) for p in BOARDS], native_pots=[xy(p) for p in POTS]), player_count=3)
            self.data.append(dict(header=header, frames=[dict(time=0., score=0., state=snapshot({k:v[lane] for k,v in native.items()}))], segments=[], diagnostics=[]))

    def append(self, actions, reward, info):
        native = {k: v[:9].cpu().tolist() for k, v in self.env.state.items()}
        acts, rewards = actions[:9].cpu().tolist(), reward[:9].cpu().tolist()
        infos = {k: v[:9].cpu().tolist() for k, v in info.items()}
        for lane, data in enumerate(self.data):
            data["segments"].append(dict(kind="step", actions=acts[lane], sparse_reward=rewards[lane], reward_known=True,
                                         shaped_reward_by_agent=[0., 0., 0.], events=[], native_info={k:v[lane] for k,v in infos.items()}))
            data["frames"].append(dict(time=len(data["segments"])/6, score=data["frames"][-1]["score"]+rewards[lane],
                                       state=snapshot({k:v[lane] for k,v in native.items()})))

    def write(self, folder):
        from better_overcooked_replay.protocol import write_replay
        return [str(write_replay(data, Path(folder) / f"{data['header']['role']}-{data['header']['style']}.ocrep.json")) for data in self.data]


def export_dual_pot_scenario(folder, device="cuda"):
    """Artificial initial state demonstrating parallel heat and cross-side meals."""
    import torch
    from overcooked_ai_py.cooperative.partition import PartitionConfig
    from .kitchen import V3PolicyKitchen
    from .controllers import make_partners
    from .evaluation import _reset_paired, ROLE_NAMES, STYLE_NAMES
    from .process_metrics import ProcessMetrics, EXTRA_NAMES
    from .experiment import save
    env = V3PolicyKitchen(PartitionConfig(layout="counter", horizon=400, recipe_rewards=(20, 10, 50)), num_envs=9, device=device)
    _reset_paired(env, 37)
    env.state["recipe"][:] = torch.tensor([0, 1, 2], device=device)
    env.state["due"][:] = 300
    env.state["mix"][:] = torch.tensor([5, 2], device=device)
    partners = make_partners(env, env.lanes//3, env.lanes%3, "parallel-v2", torch.zeros(9, device=device, dtype=torch.bool))
    partners.reset()
    recorder = EvaluationReplays(env, dict(controller_version="parallel-v2", mode="script", seed=37,
        recipe_rewards=[20, 10, 50], artificial_initial_state=True,
        initial_overrides=dict(recipe=[0, 1, 2], due=[300, 300, 300], mix=[5, 2]),
        scenario="Two pre-filled pots; heat both and deliver the first two orders on opposite sides"), ROLE_NAMES, STYLE_NAMES)
    delivered = torch.zeros(9, 2, dtype=torch.bool, device=device)
    simultaneous_tend = torch.zeros(9, dtype=torch.bool, device=device)
    simultaneous_heat = torch.zeros(9, dtype=torch.bool, device=device)
    meter = ProcessMetrics(env)
    process = torch.zeros(9, len(EXTRA_NAMES), dtype=torch.long, device=device)
    for _ in range(400):
        actions = partners.actions()
        heat = env.state["heat"].clone()
        meter.before(actions)
        _, reward, _, _, info = env.step(actions)
        simultaneous_tend |= (env.state["heat"] > heat).all(-1)
        simultaneous_heat |= (env.state["heat"] > 0).all(-1)
        for order in range(2):
            delivered[:, order] |= ((info["order_ids"] == order) & info["delivered"]).any(-1)
        process += meter.after()
        process[:, -1] += torch.as_tensor(partners.idle_actionable, device=device).sum(-1)
        recorder.append(actions, reward, info)
    if not bool(delivered.all()) or not bool(simultaneous_tend.any()):
        raise AssertionError(f"Scenario failed: delivered={delivered.tolist()}, dual_tend={simultaneous_tend.tolist()}")
    paths = recorder.write(folder)
    result = dict(first_two_orders_delivered=delivered.cpu().tolist(), simultaneous_tending=simultaneous_tend.cpu().tolist(),
                  simultaneous_nonzero_heat=simultaneous_heat.cpu().tolist(),
                  scores=[d["frames"][-1]["score"] for d in recorder.data], replay_paths=paths,
                  process_metrics={name: process[:,i].cpu().tolist() for i,name in enumerate(EXTRA_NAMES)})
    save(Path(folder) / "validation.json", result)
    return result
