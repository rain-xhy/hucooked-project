"""Self-centric dense observations from the batched v3 kitchen state."""
import torch
from torch import nn

from overcooked_ai_py.cooperative.partition.model import (
    BOARDS, CUT_POTATO, CUT_TOMATO, EMPTY, FIRST_MEAL, FLOOR, HEIGHT,
    OFFSETS, POTATO, POTS, SERVERS, SOURCES, WALL, WIDTH,
)

GROUPS = (
    ("chefs", 48), ("fixtures", 20), ("movable_items", 27), ("chopped_here", 4),
    ("local_terrain", 15), ("boards", 8), ("pots", 12), ("wall", 5),
    ("tickets", 12), ("context", 2),
)
OBS_DIM = sum(size for _, size in GROUPS)
VIEW = ((0, 1, 2), (1, 0, 2), (2, 0, 1))


def partner_action_ids(actions):
    """Map [..., 3] executed native actions to one joint partner ID per observer."""
    return torch.stack([6 * actions[..., a] + actions[..., b] for _, a, b in VIEW], dim=-1)


class V3Features(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        def constant(name, values):
            self.register_buffer(name, torch.as_tensor(values, dtype=torch.long), persistent=False)

        constant("view", VIEW)
        constant("xy", [(i % WIDTH, i // WIDTH) for i in range(WIDTH * HEIGHT)])
        constant("scale", [WIDTH, HEIGHT])
        constant("fixed", [*SOURCES.values(), *POTS, *SERVERS])
        constant("boards", BOARDS)
        constant("pots", POTS)
        constant("wall", WALL)
        constant("ports", WALL if config.layout == "counter" else WALL[1::2])
        constant("offsets", OFFSETS)
        tiles = [0 if i in FLOOR else 1 for i in range(WIDTH * HEIGHT)]
        for tile, stations in enumerate((*[(p,) for p in SOURCES.values()], BOARDS, POTS, SERVERS), 2):
            for p in stations:
                tiles[p] = tile
        for p in (WALL if config.layout == "counter" else WALL[1::2]):
            tiles[p] = 10 if config.layout == "counter" else 9
        constant("tiles", tiles)
        for n in (3, 4, 10, 11):
            constant(f"classes{n}", range(n))

    def nearest(self, own, positions, available):
        # All observers use Manhattan distance; min returns the first tie.
        delta = self.xy[positions][:, None] - own[:, :, None]
        distance = delta.abs().sum(-1).masked_fill(~available[:, None], WIDTH * HEIGHT)
        nearest = distance.argmin(-1, keepdim=True)
        offset = delta.gather(2, nearest[..., None].expand(-1, -1, -1, 2)).squeeze(2)
        present = available.any(-1)[:, None, None].expand(-1, own.shape[1], -1)
        return torch.cat((offset / self.scale * present, present), -1)

    def forward(self, s):
        pos, facing, held = s["pos"], s["facing"], s["held"]
        batch = pos.shape[0]
        def common(x):
            return x[:, None].expand(-1, 3, -1)

        own = self.xy[pos]
        peers = self.xy[pos[:, self.view[:, 1:]]]
        chefs = [own / self.scale, ((peers - own[:, :, None]) / self.scale).flatten(2),
                 (facing[:, self.view, None] == self.classes4).flatten(2),
                 (held[:, self.view, None] == self.classes10).flatten(2)]
        fixed = ((self.xy[self.fixed] - own[:, :, None]) / self.scale).flatten(2)
        fixtures = [fixed]
        for stations in (self.boards, self.ports):
            locations = stations.expand(batch, -1)
            fixtures.append(self.nearest(own, locations, torch.ones_like(locations, dtype=torch.bool))[..., :2])

        board_items = torch.where(s["cuts"] == 2,
                                  torch.where(s["board"] == POTATO, CUT_POTATO, CUT_TOMATO), s["board"])
        meals = torch.where(s["heat"] == 4, s["locked"] + FIRST_MEAL, EMPTY)
        locations = torch.cat((pos, self.boards.expand(batch, -1), self.wall.expand(batch, -1),
                               self.pots.expand(batch, -1)), -1)
        items = torch.cat((held, board_items, s["counter"], meals), -1)
        movable = [self.nearest(own, locations, items == item) for item in range(1, 10)]
        front = pos + self.offsets[facing]
        at_front = locations[:, None] == front[:, :, None]
        chopped = [held == CUT_TOMATO, held == CUT_POTATO,
                   (at_front & (items == CUT_TOMATO)[:, None]).any(-1),
                   (at_front & (items == CUT_POTATO)[:, None]).any(-1)]
        terrain = [self.tiles[pos[..., None] + self.offsets] != 0, self.tiles[front, None] == self.classes11]
        fields = [*chefs, *fixtures, *movable, torch.stack(chopped, -1), *terrain,
                  common(s["board"] / 2), common(s["cuts"] / 2)]
        fields.extend(common((s["mix"] & bit) != 0) for bit in (1, 2, 4, 8))
        fields.extend((common((s["locked"] + 1) / 3), common(s["heat"] / 4),
                       common(s["counter"] / 9), common((s["recipe"][..., None] == self.classes3).flatten(1)),
                       common((s["due"] - s["tick"]) / max(self.config.deadlines)),
                       common(s["tick"] / self.config.horizon),
                       common(torch.full_like(s["tick"], int(self.config.layout == "sync")))))
        return torch.cat(fields, dim=-1).float()
