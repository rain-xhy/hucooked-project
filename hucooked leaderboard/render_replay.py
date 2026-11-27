"""Export the winning trajectory by replaying its primitive actions in the native kitchen."""

from dataclasses import asdict
import json
from pathlib import Path
import sys

from results import score


TICK_RATE = 6
DIRECTIONS = ((0, -1), (0, 1), (1, 0), (-1, 0))
SHARED_TILES = str.maketrans({"M": "O", "L": "T", "G": "P", "K": "X", "A": "X", "B": "X"})
MIXED_TILES = str.maketrans({"1": " ", "2": " ", "3": " ", "B": "X"})
PARTITION_RECIPES = (("tomato", "egg"), ("potato",), ("potato", "tomato", "beef"))
PARTITION_ITEMS = ("", "tomato", "potato", "egg", "beef", "tomato", "potato")


def sprite(name, position, **extra):
    return {"name": name, "position": position, **extra}


def soup(position, ingredients, cooking_tick=-1, finished=False, **extra):
    return sprite("soup", position,
                  _ingredients=[sprite(kind, position) for kind in ingredients],
                  cooking_tick=cooking_tick, is_cooking=not finished and cooking_tick >= 0,
                  is_ready=finished, **extra)


def original(row):
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
    from original.simulation import Episode

    mdp = OvercookedGridworld.from_layout_name(row["layout"])
    env = Episode(mdp, OvercookedState.from_dict(row["initial_state"]), row["horizon"])

    terrain = ["".join(line) for line in mdp.terrain_mtx]
    return terrain, terrain, lambda: env.state.to_dict(), lambda actions: env.step(actions)[0]


def mixed(row):
    from overcooked_ai_py.cooperative.kitchen import MixedKitchen
    from overcooked_ai_py.cooperative.model import KitchenConfig

    env = MixedKitchen(KitchenConfig(layout=row["layout"], horizon=row.get("horizon", 300)))
    env.reset(row["seed"])

    def state():
        s = env.state
        orders = {o.id: o.kind for o in s.orders}
        orders.update({o["id"]: o["kind"] for o in s.history if "id" in o})

        def name(order_id):
            return "onion" if orders.get(order_id) == "joint" else "tomato"

        def item(token, pos):
            if token is None:
                return None
            if token.kind == "dish":
                return sprite("dish", pos)
            if token.kind == "soup":
                return soup(pos, [name(token.order_id)] * 3, finished=True, certified=token.certified)
            return sprite(name(token.order_id), pos)

        objects = [item(token, pos) for token, pos in zip(s.buffers, env.buffer_positions) if token is not None]
        for pot, pos in zip(s.pots, env.pot_positions):
            if pot.order_id is not None:
                objects.append(soup(pos, [name(pot.order_id)] * 3, max(0, 4 - pot.remaining), pot.remaining == 0))
        players = [{"position": p.position, "orientation": p.orientation, "held_object": item(p.held, p.position)}
                   for p in s.chefs]
        return {"timestep": s.tick, "players": players, "objects": objects,
                "all_orders": [{"ingredients": [name(o.id)] * 3, "id": o.id, "kind": o.kind}
                               for o in s.orders if o.status == "active"],
                "native_joint": asdict(s.joint)}

    native = ["".join(line) for line in env.terrain]
    return [r.translate(MIXED_TILES) for r in native], native, state, lambda a: env.step(a)[1]


def shared(row):
    from overcooked_ai_py.cooperative.shared import SharedKitchen, SharedKitchenConfig
    from overcooked_ai_py.cooperative.shared.layouts import LAYOUT, STATIONS

    env = SharedKitchen(SharedKitchenConfig(horizon=row.get("horizon", 300)))
    env.reset(row["seed"])

    def item(token, pos):
        if token is None:
            return None
        if token.kind == "meal":
            return soup(pos, ("onion", "tomato"), finished=True)
        names = {"main": "onion", "cooked": "onion", "side": "tomato", "prepared": "tomato",
                 "plate": "dish"}
        return sprite(names.get(token.kind, token.kind), pos, native_kind=token.kind)

    def state():
        s = env.state
        objects = [item(token, STATIONS[tile]) for token, tile in
                   ((s.cutting.item, "K"), (s.buffer, "B"), (s.assembly.plate, "A")) if token is not None]
        if s.cooker.item is not None:
            objects.append(soup(STATIONS["G"], ("onion",), max(0, 4 - s.cooker.remaining), s.cooker.remaining == 0))
        if s.assembly.main is not None:
            ingredients = ("onion", "tomato") if s.assembly.side else ("onion",)
            objects.append(soup(STATIONS["A"], ingredients, finished=True))
        return {
            "timestep": s.tick, "objects": objects,
            "players": [{"position": p.position, "orientation": p.orientation,
                         "held_object": item(p.held, p.position)} for p in s.chefs],
            "all_orders": [{"ingredients": ["onion", "tomato"], "id": o.id, "kind": o.kind}
                           for o in s.orders if o.status == "active"],
            "native_stations": {"cooker": asdict(s.cooker), "cutting": asdict(s.cutting),
                                "assembly": asdict(s.assembly), "buffer": asdict(s.buffer) if s.buffer else None},
        }

    return [line.translate(SHARED_TILES) for line in LAYOUT], list(LAYOUT), state, lambda a: env.step(a)[1]


def partition(row):
    from overcooked_ai_py.cooperative.partition import PartitionConfig, PartitionKitchen
    from overcooked_ai_py.cooperative.partition.model import (
        BOARDS, FIRST_MEAL, POTS, RECIPE_NAMES, SOURCES, WALL, WIDTH,
    )

    env = PartitionKitchen(PartitionConfig(layout=row["layout"]))
    env.reset(row["seed"])

    def position(cell):
        return int(cell % WIDTH), int(cell // WIDTH)

    def item(kind, pos):
        if kind == 0:
            return None
        if kind >= FIRST_MEAL:
            return soup(pos, PARTITION_RECIPES[kind - FIRST_MEAL], finished=True)
        return sprite(PARTITION_ITEMS[kind], pos)

    lines = ["#############", "#.....#.....#", "T.C.K.H.K.C.E", "S.....#.....S",
             "P...C.H.C...B", "#.....#.....#", "#############"]
    if row["layout"] == "counter":
        lines = [line[:6] + "=" + line[7:] if i in range(1, 6) else line for i, line in enumerate(lines)]
    aliases = str.maketrans({"#": "X", ".": " ", "P": "X", "B": "X", "E": "X",
                             "C": "X", "K": "P", "=": "X"})

    def state():
        s = env.state
        tick = int(s["tick"][0])
        players = [{"position": position(cell), "orientation": DIRECTIONS[direction],
                    "held_object": item(token, position(cell))}
                   for cell, direction, token in zip(s["pos"], s["facing"], s["held"])]
        objects = [item(kind, position(cell)) for kind, cell in SOURCES.items() if kind != 1]
        objects.extend(item(token, position(cell)) for ids, cells in (("board", BOARDS), ("counter", WALL))
                       for cell, token in zip(cells, s[ids]) if token)
        for index, pot in enumerate(POTS):
            blend, heat = int(s["mix"][index]), int(s["heat"][index])
            if blend:
                ingredients = [name for bit, name in ((1, "tomato"), (2, "potato"), (4, "egg"), (8, "beef")) if blend & bit]
                objects.append(soup(position(pot), ingredients, heat if heat else -1, heat >= 4))
        orders = [{"ingredients": PARTITION_RECIPES[recipe], "recipe": RECIPE_NAMES[recipe],
                   "id": int(order_id), "remaining": int(due) - tick}
                  for recipe, order_id, due in zip(s["recipe"], s["ids"], s["due"]) if recipe >= 0]
        return {"timestep": tick, "players": players, "objects": objects, "all_orders": orders,
                "native_progress": {k: s[k].tolist() for k in ("cuts", "heat", "locked", "due", "recipe")}}

    return [line.translate(aliases) for line in lines], lines, state, lambda a: env.step(a)[1]


def records(trajectory):
    if isinstance(trajectory, list):
        yield from trajectory
    else:
        for actions, reward, delivered, expired, order_ids in zip(
            trajectory["actions"], trajectory["reward"], trajectory["delivered"],
            trajectory["expired"], trajectory["order_ids"],
        ):
            yield {"actions": actions, "reward": reward,
                   "native_order_events": {"delivered": delivered, "expired": expired, "order_ids": order_ids}}


def save(version, revision, source):
    episode = json.loads(source.read_text())
    terrain, native, snapshot, step = {
        "original": original, "v1": mixed, "v2": shared, "v3": partition, "shared": shared,
    }[version](episode)
    episode_steps = list(records(episode["trajectory"]))
    state = snapshot()
    frames = [{"time": 0, "score": 0, "state": state}]
    segments = []
    total = 0
    for row in episode_steps:
        actions = row["actions"]
        reward = float(step(actions))
        if "reward" in row and abs(reward - row["reward"]) > 1e-6:
            raise ValueError(f"Native reward differs at transition {len(segments) + 1} in {source.name}")
        total += reward
        segments.append({"kind": "step", "actions": actions, "sparse_reward": reward,
                         "reward_known": True, "events": [], "native_events": row.get("events", []),
                         **({"native_order_events": row["native_order_events"]} if "native_order_events" in row else {})})
        frames.append({"time": len(segments) / TICK_RATE, "score": total, "state": snapshot()})
    if abs(total - score(episode)) > 1e-6:
        raise ValueError(f"Native replay score {total} differs from recorded score {score(episode)}")
    header = {"schema_version": 1, "source_version": revision, "source": "HA² leaderboard",
              "terrain": terrain, "native_terrain": native, "dynamics": "new", "mdp_params_json": "{}",
              "layout_name": f"{version} {episode.get('layout', 'shared')}", "tick_rate": TICK_RATE,
              "complete": True, "horizon": episode.get("horizon", 400 if version == "original" else 300),
              "seed": episode["seed"],
              "players": [{"id": f"Chef {i + 1}", "is_human": False} for i in range(len(state["players"]))]}
    target = source.with_name(source.name.removesuffix(".trajectory.json") + ".ocrep.json")
    target.write_text(json.dumps({"header": header, "frames": frames, "segments": segments, "diagnostics": []},
                                 separators=(",", ":")) + "\n")


def main():
    version, revision, directory = sys.argv[1:]
    for source in sorted(Path(directory).glob("*.trajectory.json")):
        save(version, revision, source)


if __name__ == "__main__":
    main()
