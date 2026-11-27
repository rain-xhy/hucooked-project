"""Continue the fixed-role BC checkpoint with reward-only hierarchical PPO."""
import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from haha.artifacts import portable_path
from haha.goal_training import mlp, tensor
from v2.policy import (ASSEMBLE, CUT, DROP_SIDE, FEATURE_DIM, GET_MAIN, GET_PLATE, GET_SIDE,
                       GOALS, SERVE, UNLOAD_WEST, USE_COOKER, SharedPolicy, ordered_chefs,
                       reserve, reserved_features, valid_actions, worker_inputs)
from v2.pure_rl import PPO
from v2.train import Execution, Trial, encode, summary

N = len(GOALS)
EXECUTION = Execution(profile="continuous", goal_lock=True, synchronize=False)
ROLE_GOALS = ({0, 1, 2, 3, 4}, {5, 6, 7}, {8, 9, 10, 11})


def expanded_menu(trial, old):
    state = trial.env.state
    menu = old.copy()
    for role, index in enumerate(ordered_chefs(trial.controller)):
        held = state.chefs[index].held
        if held is None:
            choices = [GET_MAIN, GET_PLATE, GET_SIDE, UNLOAD_WEST, USE_COOKER, CUT, ASSEMBLE, DROP_SIDE]
        else:
            choices = {"raw_main": [USE_COOKER, UNLOAD_WEST], "raw_side": [CUT],
                       "chopped_side": [DROP_SIDE, ASSEMBLE], "plate": [ASSEMBLE, DROP_SIDE],
                       "meal": [SERVE]}.get(held.kind, [])
        menu[role, choices] = True
    return menu


class Manager(nn.Module):
    """Preserve the original greedy policy and give persistent options initial exploration mass."""
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.options = mlp(FEATURE_DIM+3, N+1)
        nn.init.zeros_(self.options[-1].weight)
        nn.init.zeros_(self.options[-1].bias)
        with torch.no_grad():
            self.options[-1].bias[-1] = -2.5

    def forward(self, x):
        obs, old, extra = torch.split(x, [FEATURE_DIM+3,N,N], dim=-1)
        base = self.original(obs).masked_fill(~old.bool(), -1e9).log_softmax(-1)
        scores = self.options(obs)
        alternative = scores[..., :N].masked_fill(~extra.bool(), -1e9).log_softmax(-1)
        return torch.cat((base, alternative + scores[..., -1:]), -1)


class Batch:
    def __init__(self, seeds, opened, rng):
        self.trials = [Trial(int(s), "multi", EXECUTION) for s in seeds]
        self.rng, self.opened = rng, opened
        self.goal = np.zeros((len(seeds),3), dtype=int)
        self.left = np.zeros_like(self.goal)

    @torch.no_grad()
    def act(self, policy, sample):
        common, obs, old, _ = encode(self.trials)
        extra = np.stack([expanded_menu(t,m) for t,m in zip(self.trials,old)]) if self.opened else old
        x = np.concatenate((obs,old,extra),axis=-1)
        mask = np.concatenate((old,extra),axis=-1)
        forced = (self.left>0) & np.take_along_axis(extra,self.goal[...,None],axis=-1)[...,0]
        rows,roles = np.where(forced)
        mask[rows,roles] = False
        mask[rows,roles,self.goal[rows,roles]+N] = True
        dist = Categorical(logits=policy.manager(tensor(x)).masked_fill(~tensor(mask),-1e9))
        choices = dist.sample() if sample else dist.logits.argmax(-1)
        ids = choices.numpy()
        self.goal = ids % N
        self.left = np.where(forced,self.left,np.where(ids>=N,6,0))
        reserved = [set() for _ in self.trials]
        worker = []
        actions = np.zeros(self.goal.shape,dtype=int)
        for role in range(3):
            chefs = [ordered_chefs(t.controller)[role] for t in self.trials]
            goals = self.goal[:,role]
            wx = worker_inputs(obs[:,role],goals,np.stack([reserved_features(z) for z in reserved]))
            wm = np.stack([valid_actions(t.env,i,g,goal_lock=True) for t,i,g in zip(self.trials,chefs,goals)])
            motion = Categorical(logits=policy.worker(tensor(wx)).masked_fill(~tensor(wm),-1e9))
            a = motion.sample() if sample else motion.logits.argmax(-1)
            for j,(t,i,v) in enumerate(zip(self.trials,chefs,a.tolist())):
                actions[j,i] = v
                reserve(t.env,i,v,reserved[j])
            worker.append((wx,wm,a.numpy(),motion.log_prob(a).numpy()))
        wx,wm,a,wlp = (np.stack(z,axis=1) for z in zip(*worker))
        record = (x,mask,ids,dist.log_prob(choices).numpy()+wlp,
                  np.repeat(common[:,None,:],3,axis=1))
        return actions,record,(wx,wm,a)

    def advance(self,actions,reset):
        rewards,ends,infos = [],[],[]
        cross_steps = cross_events = 0
        for j,(trial,moves) in enumerate(zip(self.trials,actions)):
            reward,done,info = trial.step(moves)
            rewards.append([reward/20]*3)
            ends.append([done]*3)
            infos.append(info)
            for role,chef in enumerate(ordered_chefs(trial.controller)):
                events = [e for e in info["events"] if chef in e["actors"]]
                if self.goal[j,role] not in ROLE_GOALS[role]:
                    cross_steps += 1
                    cross_events += len(events)
                if events:
                    self.left[j,role]=0
            if done and reset:
                self.trials[j]=Trial(int(self.rng.integers(2**31)),"multi",EXECUTION)
                self.left[j]=0
        self.left=np.maximum(self.left-1,0)
        return np.array(rewards,np.float32),np.array(ends),infos,cross_steps,cross_events


def evaluate(policy,seeds,opened,trace=False):
    batch=Batch(seeds,opened,np.random.default_rng(0))
    history=[[] for _ in seeds]
    cross_events=0
    for tick in range(1,301):
        moves,_,_=batch.act(policy,False)
        _,_,infos,_,events=batch.advance(moves,False)
        cross_events+=events
        if trace:
            for i,info in enumerate(infos):
                history[i].append(dict(tick=tick,goals=batch.goal[i].tolist(),actions=moves[i].tolist(),
                                       reward=info["team_reward"],events=info["events"]))
    rows=[dict(seed=seed,horizon=300,trajectory=h,orders=info["orders"],**info["metrics"])
          for seed,h,info in zip(seeds,history,infos)]
    return rows,dict(**summary(rows),cross_events_per_episode=cross_events/len(rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--open-menu",action="store_true")
    args=p.parse_args()
    args.output.mkdir(parents=True)
    started=monotonic()
    def log(stage,**data):
        line=json.dumps(dict(stage=stage,seconds=round(monotonic()-started,2),**data))
        with (args.output/"progress.jsonl").open("a") as f:
            f.write(line+"\n")
        print(line,flush=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    policy=SharedPolicy()
    policy.load_state_dict(torch.load(args.checkpoint,weights_only=True,map_location="cpu"))
    policy.manager=Manager(policy.manager)
    validation=list(range(2001,2013))
    holdout=list(range(84031,84055))
    _,initial=evaluate(policy,validation,args.open_menu)
    log("initialized",**initial,seed=args.seed,open_menu=args.open_menu,checkpoint=portable_path(args.checkpoint))
    if initial["min_score"]!=420 or initial["mean_score"]!=420:
        raise ValueError("Initial policy must reproduce the 420 checkpoint")
    best=420.
    best_update=0
    torch.save(policy.state_dict(),args.output/"best.pt")
    opt=PPO(policy.manager,policy.critic,.00022,"cpu",worker=policy.worker)
    batch=Batch(rng.integers(2**31,size=12),args.open_menu,rng)
    for update in range(1,85):
        records=[]
        scores=[]
        steps=events=0
        for _ in range(128):
            moves,data,worker=batch.act(policy,True)
            rewards,ends,infos,s,e=batch.advance(moves,True)
            steps+=s
            events+=e
            scores.extend(info["metrics"]["team_score"] for done,info in zip(ends[:,0],infos) if done)
            records.append((*data,rewards,ends,*worker))
        common=encode(batch.trials)[0]
        opt.update(records,np.repeat(common[:,None,:],3,axis=1))
        log("ppo_update",update=update,environment_steps=1536*update,sampled_scores=scores,
            cross_steps=steps,cross_events=events)
        if update%6==0:
            _,val=evaluate(policy,validation,args.open_menu)
            log("validation",update=update,**val)
            if val["mean_score"]>best:
                best,best_update=val["mean_score"],update
                torch.save(policy.state_dict(),args.output/"best.pt")
    torch.save(policy.state_dict(),args.output/"last.pt")
    for label in ("last","best"):
        if label=="best":
            policy.load_state_dict(torch.load(args.output/"best.pt",weights_only=True))
        rows,stats=evaluate(policy,holdout,args.open_menu,trace=True)
        import gzip
        with gzip.open(args.output/f"{label}-evaluation.json.gz","wt") as f:
            json.dump(rows,f,separators=(",",":"))
        log(label,update=84 if label=="last" else best_update,**stats)

if __name__=="__main__":
    main()
