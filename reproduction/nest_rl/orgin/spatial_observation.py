"""Compact NestRL observations decoded into semantic maps by a trainable CNN."""

import torch
from torch import nn

from torch_kitchen import FOOD_NAMES, HEIGHT, PLATE, WIDTH


MAP_CHANNELS = 33
PLAYER_FEATURES = 16
GRID_CELLS = HEIGHT * WIDTH
OBSERVATION_SIZE = GRID_CELLS + 3 * PLATE + 7
TILE_CLASSES = ("Floor", "Counter", "CutBoard", "DeliverSquare")


def pack_views(tiles, objects, chopped, plated, positions, orientations, actions):
    """Return [kitchen, perspective, feature], with the observing chef first."""
    common = torch.cat((tiles.expand(len(positions), -1), objects, chopped, plated), dim=-1)
    chefs = torch.stack((positions, orientations, actions), dim=-1)
    views = torch.stack((chefs, chefs.flip(1)), dim=1).flatten(2)
    return torch.cat((common[:, None, :].expand(-1, 2, -1), views), dim=-1).float()


def pack_kitchen(kitchen, actions=None):
    if actions is None:
        actions = torch.full_like(kitchen.agent_pos, -1)
    return pack_views(kitchen.tiles, kitchen.object_pos, kitchen.chopped, kitchen.plated,
                      kitchen.agent_pos, kitchen.orientation, actions)


def world_tiles(world):
    if (world.width, world.height) != (WIDTH, HEIGHT):
        raise ValueError(f"Spatial NestRL expects a {WIDTH}x{HEIGHT} kitchen")
    tiles = torch.full((GRID_CELLS,), -1, dtype=torch.long)
    for tile_id, name in enumerate(TILE_CLASSES):
        for obj in world.world_objects[name]:
            x, y = obj.location
            tiles[y * WIDTH + x] = tile_id
    if (tiles < 0).any():
        raise ValueError("Spatial NestRL received an unsupported terrain type")
    return tiles


def pack_world(world, tiles, actions=(-1, -1)):
    """Read the same fields from the original CPU simulation."""
    foods = [world.world_objects[name][0] for name in FOOD_NAMES]
    plate = world.world_objects["Plate"][0]

    def location(obj):
        x, y = obj.location
        return y * WIDTH + x

    return pack_views(
        tiles, torch.tensor([[location(obj) for obj in (*foods, plate)]]),
        torch.tensor([[obj.done() for obj in foods]]),
        torch.tensor([[any(obj is food for obj in plate.content) for food in foods]]),
        torch.tensor([[location(chef) for chef in world.agents]]),
        torch.tensor([[chef.orientation for chef in world.agents]]), torch.tensor([actions]),
    )[0].numpy()


def unpack_observation(obs):
    """Return the 33 spatial planes and the two chefs' coordinates/actions."""
    tiles, objects, chopped, plated, chefs = obs.split((GRID_CELLS, PLATE + 1, PLATE, PLATE, 6), dim=-1)
    positions, orientations, actions = chefs.unflatten(-1, (2, 3)).long().unbind(-1)
    cells = torch.arange(GRID_CELLS, device=obs.device)
    terrain = tiles[..., None, :] == torch.arange(len(TILE_CLASSES), device=obs.device)[:, None]
    items = objects[..., :, None] == cells
    food, plate = items.split((PLATE, 1), dim=-2)
    chefs_at = positions[..., :, None] == cells
    facing = orientations[..., :, None] == torch.arange(1, 5, device=obs.device)
    heading = (chefs_at[..., :, None, :] & facing[..., None]).flatten(-3, -2)
    grid = torch.cat((terrain, food, food & chopped[..., None].bool(), plate,
                      food & plated[..., None].bool(), chefs_at, heading), dim=-2)

    x = (positions % WIDTH) / WIDTH
    y = (positions // WIDTH) / HEIGHT
    xy = torch.stack((x, y), dim=-1).flatten(-2)
    previous = (actions[..., :, None] == torch.arange(6, device=obs.device)).flatten(-2)
    return grid.unflatten(-1, (HEIGHT, WIDTH)).to(obs.dtype), torch.cat((xy, previous), dim=-1).to(obs.dtype)


class SpatialFeatures(nn.Module):
    observation_size = OBSERVATION_SIZE
    out_dim = 128 + PLAYER_FEATURES

    def __init__(self):
        super().__init__()
        self.map_net = nn.Sequential(
            nn.Conv2d(MAP_CHANNELS, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(16 * GRID_CELLS, 128), nn.ReLU(),
        )

    def forward(self, inputs):
        # The upstream latent critic appends its latent before calling this module.
        obs, tail = inputs[..., :OBSERVATION_SIZE], inputs[..., OBSERVATION_SIZE:]
        grid, players = unpack_observation(obs)
        maps = self.map_net(grid.reshape(-1, MAP_CHANNELS, HEIGHT, WIDTH))
        maps = maps.reshape(*inputs.shape[:-1], 128)
        return torch.cat((maps, players, tail), dim=-1)


def make_spatial_policy(**kwargs):
    from learning.model import LatentPolicy

    if kwargs["obs_shape"] != (OBSERVATION_SIZE,):
        raise ValueError(f"Unexpected spatial observation shape: {kwargs['obs_shape']}")
    # Build PPO's existing MLPs for the feature size after the CNN, then attach
    # a spatial stem at each point where upstream consumes observations.
    policy = LatentPolicy(**dict(kwargs, obs_shape=(SpatialFeatures.out_dim,)))
    policy.actor.cnn = SpatialFeatures()
    policy.critic.cnn = SpatialFeatures()
    policy.encoder.cnn = SpatialFeatures()
    return policy
