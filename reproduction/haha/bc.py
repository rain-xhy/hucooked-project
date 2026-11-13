"""Human-data preparation, BC partners and held-out human proxies."""

import ast
import json
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import Action, OvercookedGridworld, OvercookedState

from . import ENVIRONMENT_COMMIT, UPSTREAM_COMMIT
from .artifacts import atomic_path, read_json, save_torch, write_json
from .encoding import bc_features, make_planner

DATA_URL = (
    "https://raw.githubusercontent.com/StephAO/HAHA/"
    f"{UPSTREAM_COMMIT}/data/2019_hh_trials_all.pickle"
)


def prepare(dataset, output, layouts, seed):
    dataset, output = Path(dataset), Path(output)
    if not dataset.exists():
        dataset.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "HAHA-reproduction"})
        with atomic_path(dataset) as part:
            with urllib.request.urlopen(request, timeout=120) as src, part.open("wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
            with part.open("rb") as src:
                if src.read(64).startswith(b"version https://git-lfs"):
                    raise ValueError("Downloaded an LFS pointer instead of the human dataset")
    # The source is the pinned author's trusted pickle. Never use untrusted pickles.
    frame = pd.read_pickle(dataset).copy()
    required = {"layout_name", "state", "joint_action"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing dataset columns: {required - set(frame.columns)}")
    frame["layout_name"] = frame["layout_name"].replace({
        "random0": "forced_coordination", "random3": "counter_circuit_o_1order"
    })
    group_column = next((c for c in ("trial_id", "episode_id") if c in frame), None)
    if group_column is None:
        raise ValueError("A trajectory id is required; refusing a leaking row-level split")
    for layout in layouts:
        target = output / f"{layout}.npz"
        spec = {"schema": 1, "layout": layout, "upstream_commit": UPSTREAM_COMMIT,
                "environment_commit": ENVIRONMENT_COMMIT,
                "dataset_bytes": dataset.stat().st_size, "features": 96,
                "split_unit": "trajectory"}
        if target.exists():
            with np.load(target, allow_pickle=False) as data:
                if "source" not in data or json.loads(str(data["source"])) != spec:
                    raise ValueError(f"Prepared data source differs: {target}; use a new run directory")
            print(f"already prepared: {layout}", flush=True)
            continue
        rows = frame[frame.layout_name == layout]
        if rows.empty:
            raise ValueError(f"No human trajectories for {layout}")
        if rows[group_column].isna().any():
            raise ValueError(f"Missing trajectory ids in {layout}")
        mdp = OvercookedGridworld.from_layout_name(layout)
        planner = make_planner(mdp)
        observations, labels, groups = [], [], []
        for _, row in rows.iterrows():
            joint = row["joint_action"]
            if isinstance(joint, str):
                try:
                    joint = json.loads(joint)
                except json.JSONDecodeError:
                    joint = ast.literal_eval(joint)
            actions = [tuple(a) if isinstance(a, (list, tuple, np.ndarray))
                       else a.lower() if isinstance(a, str) else a for a in joint]
            if len(actions) != 2:
                raise ValueError("Expected a joint action for two players")
            if all(a == Action.STAY for a in actions):
                continue
            state = row["state"]
            if isinstance(state, str):
                state = json.loads(state)
            if isinstance(state, dict):
                state = OvercookedState.from_dict(state)
            for player in range(2):
                observations.append(bc_features(mdp, state, planner, player))
                labels.append(Action.ACTION_TO_INDEX[actions[player]])
                groups.append(str(row[group_column]))
        unique = np.unique(groups)
        if len(unique) < 4:
            raise ValueError(f"At least four independent trajectories needed for {layout}")
        output.mkdir(parents=True, exist_ok=True)
        with atomic_path(target) as temporary:
            np.savez_compressed(temporary, observations=np.asarray(observations, dtype=np.float32),
                                actions=np.asarray(labels, dtype=np.int64), groups=np.asarray(groups),
                                source=json.dumps(spec))
        print(f"prepared {layout}: {len(labels)} player actions, {len(unique)} trajectories", flush=True)
    write_json(output / "source.json", {
        "url": DATA_URL, "dataset": str(dataset.resolve()), "seed": seed,
        "layouts": layouts, "split_unit": "trajectory",
        "note": "Trajectory-level split, rather than the upstream row-level random_split.",
    })


class BCNetwork(nn.Module):
    def __init__(self, input_dim=96, hidden=64):
        super().__init__()
        self.input_dim, self.hidden = input_dim, hidden
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 6))
        # Match the upstream BC initializer: Linear weights keep PyTorch defaults,
        # while all biases are zeroed.
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.bias)

    def forward(self, observation):
        return self.layers(observation)


class BCPartner:
    def __init__(self, network, device="cpu"):
        self.device = torch.device(device)
        self.network = network.to(self.device).eval()
        self.network.requires_grad_(False)
        self._cache_context = (None, None)
        self._probability_cache = {}

    @classmethod
    def load(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = BCNetwork(checkpoint["input_dim"], checkpoint["hidden"])
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model, device)

    @staticmethod
    def _object_key(obj):
        if obj is None:
            return None
        if obj.name == "soup":
            # The upstream equality omits cook_time. Both fields affect features.
            return (obj.name, obj.position, tuple(obj.ingredients),
                    obj._cooking_tick, obj._cook_time)
        return obj.name, obj.position

    def _probabilities(self, mdp, state, planner, player):
        cached_mdp, cached_planner = self._cache_context
        if mdp is not cached_mdp or planner is not cached_planner:
            self._cache_context = (mdp, planner)
            self._probability_cache.clear()
        # The pinned BC encoder ignores timestep. Preserve counter insertion
        # order because it can break ties between equally close objects.
        key = (
            player,
            tuple((p.pos_and_or, self._object_key(p.held_object)) for p in state.players),
            tuple((pos, self._object_key(obj)) for pos, obj in state.objects.items()),
        )
        probs = self._probability_cache.get(key)
        if probs is not None:
            return probs
        obs = torch.as_tensor(bc_features(mdp, state, planner, player), device=self.device)
        with torch.no_grad():
            probs = self.network(obs).softmax(-1).cpu().numpy()
        if len(self._probability_cache) >= 8192:
            self._probability_cache.clear()
        self._probability_cache[key] = probs
        return probs

    def act(self, mdp, state, planner, player, rng, deterministic=False):
        probs = self._probabilities(mdp, state, planner, player)
        return int(probs.argmax() if deterministic else rng.choice(6, p=probs / probs.sum()))


class RandomPartner:
    def act(self, mdp, state, planner, player, rng, deterministic=False):
        return int(rng.integers(6))


def selfplay_score(network, layout, episodes, horizon, seed, device, *, tensor=False):
    if tensor:
        from .tensor_features import TensorFeatures
        from .tensor_kitchen import TensorKitchen

        kitchen = TensorKitchen((layout,) * episodes, device=device, horizon=horizon)
        features = TensorFeatures(kitchen)
        generator = torch.Generator(device=device).manual_seed(seed)
        state = kitchen.initial
        returns = torch.zeros(episodes, device=device)
        with torch.no_grad():
            for _ in range(horizon):
                probabilities = network(features.bc_observation(state)).softmax(-1)
                joint = torch.multinomial(probabilities.reshape(-1, 6), 1, generator=generator)
                result = kitchen.step(state, joint.reshape(episodes, 2))
                state = result.state
                returns += result.reward
        return float(returns.mean())

    partner = BCPartner(network, device)
    mdp = OvercookedGridworld.from_layout_name(layout)
    planner = make_planner(mdp)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon)
    rng = np.random.default_rng(seed)
    returns = []
    for _ in range(episodes):
        env.reset()
        total = 0.0
        for _ in range(horizon):
            joint = [Action.INDEX_TO_ACTION[partner.act(mdp, env.state, planner, i, rng)]
                     for i in range(2)]
            _, reward, done, _ = env.step(joint)
            total += float(reward)
            if done:
                break
        returns.append(total)
    return float(np.mean(returns))


def save_network(path, network, **metadata):
    save_torch(path, {"input_dim": network.input_dim, "hidden": network.hidden,
                     "state_dict": network.state_dict(), **metadata})


def _prepare_bc_layout(args, layout, index):
    root = args.run_dir / "bc"
    directory = root / layout
    directory.mkdir(parents=True, exist_ok=True)
    with np.load(args.run_dir / "data" / f"{layout}.npz", allow_pickle=False) as data:
        obs, actions, groups = data["observations"], data["actions"], data["groups"]
    split_path = directory / "split.json"
    if split_path.exists():
        split_info = read_json(split_path)
    else:
        ids = np.random.default_rng(args.seed + index).permutation(np.unique(groups))
        if len(ids) < 4:
            raise ValueError(f"At least four independent trajectories needed for {layout}")
        partitions = []
        for candidate, half in enumerate(np.array_split(ids, 2)):
            count = max(1, int(len(half) * .1))
            partitions.append({"candidate": candidate, "train": half[count:].tolist(),
                               "validation": half[:count].tolist()})
        split_info = {"seed": args.seed + index, "unit": "trajectory",
                      "partitions": partitions, "checkpoint_selection": "selfplay_return"}
        write_json(split_path, split_info)
    split_info["roles_ready"] = False
    write_json(split_path, split_info)
    return directory, split_info, (obs, actions, groups)

def _train_bc_candidate(args, device, layout, directory, dataset, partition, seed, candidate_deadline,
                        model=None):
    obs, actions, groups = dataset
    candidate = partition["candidate"]
    train_mask = np.isin(groups, partition["train"])
    val_mask = np.isin(groups, partition["validation"])
    if not train_mask.any() or not val_mask.any() or np.any(train_mask & val_mask):
        raise ValueError(f"Invalid trajectory partition in {layout}")
    mixed = args.environment == "mixed"
    if mixed:
        from .mixed_data import mixed_selfplay_score
        from .mixed_kitchen import TensorMixedKitchen
        batch_rng = torch.Generator(device=device).manual_seed(seed)
    else:
        torch.manual_seed(seed)
        model = BCNetwork(obs.shape[1], args.bc_hidden)
    model.to(device)
    fused = mixed and device == "cuda"
    optimizer = torch.optim.Adam(model.parameters(), lr=args.bc_lr, fused=True if fused else None)
    frequencies = np.bincount(actions[train_mask], minlength=6) + 1
    weights = 1.0 / frequencies
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(
        6 * weights / weights.sum(), dtype=torch.float32, device=device
    ))
    train_x = torch.from_numpy(obs[train_mask]).to(device)
    train_y = torch.from_numpy(actions[train_mask]).to(device)
    loader = None if mixed else DataLoader(torch.arange(len(train_y)), batch_size=args.bc_batch_size,
                                           shuffle=True, num_workers=0)
    val_x = torch.from_numpy(obs[val_mask]).to(device)
    val_y = torch.from_numpy(actions[val_mask]).to(device)
    candidate_file = directory / f"candidate_{candidate}.pt"
    resume_file = directory / f"candidate_{candidate}_last.pt"
    first_epoch, best_score = 1, float("-inf")
    if resume_file.exists():
        saved = torch.load(resume_file, map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer_state = saved["optimizer"]
        if mixed:
            for group in optimizer_state["param_groups"]:
                group.update(fused=True if fused else None, foreach=False if fused else None)
        optimizer.load_state_dict(optimizer_state)
        first_epoch, best_score = saved["epoch"] + 1, saved["best_score"]
        if mixed:
            if "batch_rng" in saved:
                if saved.get("batch_rng_device", device) == device:
                    batch_rng.set_state(saved["batch_rng"].cpu())
            elif device == "cuda" and saved["cuda_rng"] is not None:
                batch_rng.set_state(saved["cuda_rng"][torch.cuda.current_device()].cpu())
            elif device == "cpu":
                batch_rng.set_state(saved["torch_rng"].cpu())
        else:
            torch.set_rng_state(saved["torch_rng"].cpu())
            if device == "cuda" and saved["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    if candidate_file.exists():
        selected = torch.load(candidate_file, map_location="cpu", weights_only=True)
        best_score = max(best_score, selected["selfplay_return"])
    def save_resume(epoch):
        rng = {"batch_rng": batch_rng.get_state(), "batch_rng_device": device} if mixed else {
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
        }
        save_torch(resume_file, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                 "epoch": epoch, "best_score": best_score, **rng})
    if not resume_file.exists():
        save_resume(0)
    kitchen = None
    log_path = directory / f"candidate_{candidate}.csv"
    with log_path.open("a", encoding="utf-8") as log:
        if log.tell() == 0:
            log.write("epoch,train_loss,validation_loss,selfplay_return\n")
        for epoch in range(first_epoch, args.epochs + 1):
            if time.monotonic() >= candidate_deadline:
                break
            model.train().requires_grad_(True)
            loss_sum = torch.zeros((), device=device, dtype=torch.float64)
            epoch_order = (torch.randperm(len(train_y), device=device, generator=batch_rng) if mixed else
                           torch.cat(list(loader)).to(device))
            incomplete = False
            for indices in epoch_order.split(args.bc_batch_size):
                if time.monotonic() >= candidate_deadline:
                    incomplete = True
                    break
                x, y = train_x[indices], train_y[indices]
                loss = criterion(model(x), y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += loss.detach().double() * len(y)
            if incomplete:
                # Keep the last complete epoch, including its optimizer/RNG.
                break
            model.eval()
            with torch.no_grad():
                losses = torch.stack((loss_sum / len(train_y), criterion(model(val_x), val_y).double()))
                train_loss, val_loss = losses.tolist()
            score = None
            if epoch == 1 or epoch % args.bc_eval_every == 0 or epoch == args.epochs:
                if mixed:
                    if kitchen is None:
                        kitchen = TensorMixedKitchen((layout,) * args.selection_episodes, args.horizon, device=device)
                    score = mixed_selfplay_score(model, kitchen, args.seed + 10000)
                else:
                    score = selfplay_score(model, layout, args.selection_episodes,
                                           args.horizon, args.seed + 10000, device,
                                           tensor=args.vec_env == "tensor-native")
                if score > best_score:
                    best_score = score
                    save_network(candidate_file, model, epoch=epoch,
                                 selfplay_return=score, artifact_id=uuid.uuid4().hex)
            save_resume(epoch)
            log.write(f"{epoch},{train_loss},{val_loss},{score if score is not None else ''}\n")
            log.flush()
    if not candidate_file.exists():
        print(f"No complete selected BC epoch for {layout}, candidate {candidate}; resume BC", flush=True)
        return None
    selected = torch.load(candidate_file, map_location="cpu", weights_only=True)
    return selected["selfplay_return"]

def _publish_bc_roles(args, layout, directory, split_info, best_scores):
    mixed = args.environment == "mixed"
    if mixed:
        from .mixed_tasks import MIXED_COMMIT
    split_path = directory / "split.json"
    # Use the weaker model for training, stronger as the held-out proxy.
    proxy = int(np.argmax(best_scores))
    bc = 1 - proxy
    proxy_role = "expert_proxy" if mixed else "human_proxy"
    for role, candidate in (("bc", bc), (proxy_role, proxy)):
        checkpoint = torch.load(directory / f"candidate_{candidate}.pt",
                                map_location="cpu", weights_only=True)
        checkpoint.update(role=role, candidate=candidate, layout=layout)
        if mixed:
            checkpoint.update(environment_commit=MIXED_COMMIT, horizon=args.horizon)
        save_torch(directory / f"{role}.pt", checkpoint)
    split_info.update(candidate_selfplay_returns=best_scores, bc_candidate=bc,
                      proxy_candidate=proxy, roles_ready=True)
    write_json(split_path, split_info)
    if max(best_scores) <= 0:
        print(f"WARNING: both BC candidates have non-positive self-play returns on {layout}", flush=True)

def _run_bc_stream(args, device, layout, directory, dataset, partition, seed, deadline, model):
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        return _train_bc_candidate(args, device, layout, directory, dataset, partition, seed, deadline, model)


def train_bc(args, device):
    deadline = time.monotonic() + args.max_hours * 3600
    mixed = args.environment == "mixed"
    if mixed and device == "cuda":
        jobs = []
        # Construct models on the host before starting threads. Thereafter each
        # candidate owns its CUDA stream, optimizer, batch RNG, and output files.
        for index, layout in enumerate(args.layouts):
            directory, split, dataset = _prepare_bc_layout(args, layout, index)
            models = []
            for partition in split["partitions"]:
                seed = args.seed + index * 2 + partition["candidate"]
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(seed)
                    model = BCNetwork(dataset[0].shape[1], args.bc_hidden)
                models.append((partition, seed, model))
            jobs.append((layout, directory, split, dataset, models))
        with ThreadPoolExecutor(max_workers=sum(len(job[-1]) for job in jobs)) as pool:
            pending = [
                (layout, directory, split,
                 [pool.submit(_run_bc_stream, args, device, layout, directory, dataset, partition, seed,
                              deadline, model) for partition, seed, model in models])
                for layout, directory, split, dataset, models in jobs
            ]
            for layout, directory, split, futures in pending:
                scores = [future.result() for future in futures]
                if all(score is not None for score in scores):
                    _publish_bc_roles(args, layout, directory, split, scores)
        return

    remaining = 2 * len(args.layouts)
    for index, layout in enumerate(args.layouts):
        directory, split, dataset = _prepare_bc_layout(args, layout, index)
        scores = []
        for partition in split["partitions"]:
            seed = args.seed + index * 2 + partition["candidate"]
            candidate_deadline = time.monotonic() + max(0, deadline - time.monotonic()) / remaining
            remaining -= 1
            model = None
            if mixed:
                torch.manual_seed(seed)
                model = BCNetwork(dataset[0].shape[1], args.bc_hidden)
            score = _train_bc_candidate(args, device, layout, directory, dataset, partition, seed,
                                        candidate_deadline, model)
            if score is None:
                return
            scores.append(score)
        _publish_bc_roles(args, layout, directory, split, scores)
