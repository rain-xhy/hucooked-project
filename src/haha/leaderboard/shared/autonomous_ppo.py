"""Online primitive-action PPO in SharedKitchen, initialized only from the 420 checkpoint."""
import argparse
import ctypes
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from haha.artifacts import portable_path
from v2.policy import SharedPolicy, ordered_chefs
from v2.train import Execution, Trial, act, encode

DEVICE = "cuda"
HORIZON = 300
EXEC = Execution(profile="continuous", goal_lock=True)
START = [23, 18, 19, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0]
ITEMS = [None, "raw_main", "raw_side", "chopped_side", "plate", "meal"]
DIRECTIONS = [(0,-1), (0,1), (1,0), (-1,0)]


def state_row(t):
    s = t.env.state
    chefs = [s.chefs[i] for i in ordered_chefs(t.controller)]
    def kind(item):
        return ITEMS.index(item.kind if item else None)
    def ticket(item):
        return item.order_id if item else -1
    cuts = 1 + s.cutting.progress if s.cutting.item else 0
    return ([c.position[0] + 7*c.position[1] for c in chefs] +
            [DIRECTIONS.index(c.orientation) for c in chefs] +
            [kind(c.held) for c in chefs] +
            [ticket(s.cooker.item), s.cooker.remaining, cuts, int(s.assembly.side is not None),
             kind(s.assembly.plate), ticket(s.assembly.main), kind(s.buffer),
             sum(o.started_at >= 0 for o in s.orders), sum(o.status == "served" for o in s.orders), s.tick])


class Kitchen:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(library.resolve()))
        self.fn = self.lib.shared_step
        self.fn.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int, ctypes.c_void_p]
        self.fn.restype = None

    def step(self, s, a):
        a = a.contiguous()
        r = torch.empty(len(s), device=DEVICE)
        self.fn(s.data_ptr(), a.data_ptr(), r.data_ptr(), len(s), torch.cuda.current_stream().cuda_stream)
        return r


def fresh(n):
    return torch.tensor(START, dtype=torch.int32, device=DEVICE).repeat(n, 1)


def observe(s):
    # Fixed one-hot layout lets the network learn locations and interactions from data.
    ids = torch.cat((s[:,:3], s[:,3:6], s[:,6:9], (s[:,9:10]+1).clamp_max(1),
                     s[:,9:10].remainder(2), s[:,10:14], (s[:,14:15]+1).clamp_max(1),
                     s[:,15:16], (s[:,16:17]-s[:,17:18]), s[:,16:17].remainder(2)),1)
    offsets = observe.offsets
    out = torch.zeros((len(s), observe.width+1), device=DEVICE)
    out.scatter_(1, ids.long()+offsets, 1.)
    out[:,-1] = s[:,-1]/HORIZON
    return out


dims = [49]*3+[4]*3+[6]*3+[2,2,5,4,2,6,2,6,3,2]
observe.width = sum(dims)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(observe.width+1,192), nn.Tanh(), nn.Linear(192,192),nn.Tanh())
        self.actor = nn.Linear(192,18)
        self.critic = nn.Linear(192,1)

    def forward(self, s):
        h = self.body(observe(s))
        return self.actor(h).reshape(-1,3,6), self.critic(h).flatten()


def seeds_and_parity(old, kitchen, n, rng):
    trials = [Trial(int(i),"multi",EXEC) for i in rng.integers(2**31,size=n)]
    sim = fresh(n)
    samples, labels = [], []
    for tick in range(HORIZON):
        _, obs, masks, learns = encode(trials)
        moves = act(trials,old,obs,masks,learns,None)[0]
        targets = np.stack([m[list(ordered_chefs(t.controller))] for t,m in zip(trials,moves)])
        samples.append(sim.clone())
        labels.append(torch.as_tensor(targets,device=DEVICE))
        # Leave one clean lane and exercise off-trajectory transition parity in the others.
        noise = rng.random((n,3)) < np.linspace(0,.12,n)[:,None]
        executed = np.where(noise,rng.integers(6,size=(n,3)),targets)
        actual = []
        for t, canonical in zip(trials,executed):
            physical = np.empty(3,dtype=int)
            physical[list(ordered_chefs(t.controller))] = canonical
            actual.append(t.step(physical)[0])
        predicted = kitchen.step(sim,torch.tensor(executed,device=DEVICE))
        expected = np.array([state_row(t) for t in trials])
        if not np.array_equal(expected,sim.cpu().numpy()) or not np.array_equal(actual,predicted.cpu().numpy()):
            raise RuntimeError(f"Native/CUDA mismatch at initial collection tick {tick+1}")
    assert trials[0].reward == 420
    return torch.cat(samples),torch.cat(labels)


@torch.no_grad()
def greedy(policy,kitchen):
    s=fresh(1)
    for _ in range(HORIZON):
        a=policy(s)[0].argmax(-1)
        kitchen.step(s,a)
    return int(s[0,17].item())*20


@torch.no_grad()
def official(policy,seeds):
    trials=[Trial(seed,"multi",EXEC) for seed in seeds]
    histories=[[] for _ in trials]
    for tick in range(HORIZON):
        obs=torch.tensor([state_row(t) for t in trials],device=DEVICE)
        canonical=policy(obs)[0].argmax(-1).cpu().numpy()
        for i,(t,actions) in enumerate(zip(trials,canonical)):
            physical=np.empty(3,dtype=int)
            physical[list(ordered_chefs(t.controller))]=actions
            reward,_,info=t.step(physical)
            histories[i].append(dict(tick=tick+1,actions=physical.tolist(),events=info["events"],reward=reward))
    return [dict(seed=seed,horizon=HORIZON,trajectory=h,orders=t.env._info([],0)["orders"],
                 **t.env.episode_metrics()) for seed,h,t in zip(seeds,histories,trials)]


def train(args):
    rng=np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    observe.offsets=torch.as_tensor(np.cumsum([0]+dims[:-1]),device=DEVICE)
    kitchen=Kitchen(args.library)
    out=args.output
    out.mkdir(parents=True)
    def log(**row):
        print(json.dumps(row),flush=True)
        with (out/"progress.jsonl").open("a") as f:
            f.write(json.dumps(row)+"\n")
    old=SharedPolicy()
    old.load_state_dict(torch.load(args.checkpoint,map_location="cpu",weights_only=True))
    states,targets=seeds_and_parity(old,kitchen,32,rng)
    log(stage="parity",native_transitions=len(targets),checkpoint=portable_path(args.checkpoint))
    policy=Policy().to(DEVICE)
    opt=torch.optim.Adam(policy.parameters(),lr=.002)
    for update in range(1,4001):
        ix=torch.randint(len(states),(1024,),device=DEVICE)
        logits,_=policy(states[ix])
        loss=F.cross_entropy(logits.reshape(-1,6),targets[ix].flatten())
        opt.zero_grad();loss.backward();opt.step()
        if update%250==0:
            score=greedy(policy,kitchen)
            log(stage="initialization",update=update,score=score)
            if score==420: break
    if score!=420: raise ValueError("Could not recover the 420 starting policy")
    torch.save(policy.state_dict(),out/"start.pt")
    n=args.lanes
    s=fresh(n)
    opt=torch.optim.Adam(policy.parameters(),lr=args.lr)
    best=score
    def verify():
        rows=official(policy,range(84031,84055))
        scores=[r["team_score"] for r in rows]
        with gzip.open(out/"official-evaluation.json.gz","wt") as f: json.dump(rows,f)
        log(stage="native_test",scores=scores)
        return min(scores)>=440
    observations=torch.empty((HORIZON,n,len(START)),device=DEVICE,dtype=torch.int32)
    actions=torch.empty((HORIZON,n,3),device=DEVICE,dtype=torch.int64)
    old_lp=torch.empty((HORIZON,n),device=DEVICE)
    rewards=torch.empty_like(old_lp); values=torch.empty_like(old_lp)
    advantages=torch.empty_like(old_lp)
    for epoch in range(1,args.epochs+1):
        s=fresh(n)
        with torch.no_grad():
            for k in range(HORIZON):
                observations[k].copy_(s)
                logits,v=policy(s)
                dist=torch.log_softmax(logits/args.temperature,-1)
                a=(dist-torch.empty_like(dist).exponential_().log()).argmax(-1)
                actions[k]=a; values[k]=v; old_lp[k]=dist.gather(-1,a.unsqueeze(-1)).squeeze(-1).sum(-1)
                rewards[k]=kitchen.step(s,a)/20
            gae=next_v=0
            for k in reversed(range(HORIZON)):
                gae=rewards[k]+.997*next_v-values[k]+.997*.95*gae
                advantages[k]=gae
                next_v=values[k]
            returns=(advantages+values).flatten()
            adv=advantages.flatten()
            adv=(adv-adv.mean())/(adv.std()+1e-6)
        for _ in range(3):
            for ix in torch.randperm(n*HORIZON,device=DEVICE).split(4096):
                logits,v=policy(observations.flatten(0,1)[ix])
                lp=F.log_softmax(logits/args.temperature,-1)
                chosen=actions.flatten(0,1)[ix]
                ratio=(lp.gather(-1,chosen.unsqueeze(-1)).squeeze(-1).sum(-1)-old_lp.flatten()[ix]).exp()
                obj=torch.minimum(ratio*adv[ix],ratio.clamp(.85,1.15)*adv[ix]).mean()
                entropy=-(lp.exp()*lp).sum(-1).mean()
                loss=-obj+.25*F.smooth_l1_loss(v,returns[ix])-.003*entropy
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),.8);opt.step()
        score=greedy(policy,kitchen)
        log(stage="online_ppo",epoch=epoch,env_steps=epoch*n*HORIZON,greedy=score,
            sampled_mean=float((s[:,17].float()*20).mean()),sampled_max=int(s[:,17].max())*20)
        if score>best:
            best=score;torch.save(policy.state_dict(),out/"best.pt")
            if score>=440 and verify(): break
    torch.save(policy.state_dict(),out/"last.pt")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--lanes",type=int,default=512)
    p.add_argument("--epochs",type=int,default=45)
    p.add_argument("--temperature",type=float,default=1.35)
    p.add_argument("--lr",type=float,default=.00015)
    p.add_argument("--seed",type=int,default=413)
    train(p.parse_args())
