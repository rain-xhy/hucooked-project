"""Explore persistent, one-to-one chef assignments from the 420-point BC checkpoint."""
import argparse
from collections import Counter
from copy import copy
import gzip
from itertools import permutations
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from haha.artifacts import portable_path
from haha.goal_training import mlp, tensor
from shared.transfer_ppo import Batch, Manager
from v2.policy import FEATURE_DIM, SharedPolicy, features, ordered_chefs
from v2.pure_rl import PPO
from v2.train import encode, summary

ASSIGNMENTS = np.array(list(permutations(range(3))))
IDENTITY = np.eye(6, dtype=np.float32)
INPUTS = FEATURE_DIM + 7
STAGES = {"load", "unload", "assemble", "serve"}
WORK = {"load", "unload", "assemble", "serve", "take_source", "place_assembly",
        "take_meal", "take_cutting", "chop", "place_cutting"}


class TeamBatch:
    def __init__(self, seeds, rng):
        self.low = Batch(seeds, False, rng)
        self.home = [copy(t.controller) for t in self.low.trials]
        self.assignment = np.zeros(len(seeds), dtype=int)
        self.age = np.zeros(len(seeds), dtype=int)
        self.ready = np.ones(len(seeds), dtype=bool)
        self.spell = np.zeros(len(seeds), dtype=int)
        self.participants = [set() for _ in seeds]
        self.metrics = Counter()

    def inputs(self):
        return np.stack([np.concatenate((features(t.env, home), IDENTITY[a],
                                         np.array([age/16], dtype=np.float32)))
                         for t,home,a,age in zip(self.low.trials,self.home,self.assignment,self.age)])

    def end_spell(self, i):
        if self.assignment[i] and self.spell[i]:
            self.metrics["finished_nonidentity_spells"] += 1
            self.metrics["spell_ticks"] += int(self.spell[i])
            self.metrics["two_chefs_worked"] += int(len(self.participants[i]) >= 2)
        self.spell[i] = 0
        self.participants[i].clear()

    @torch.no_grad()
    def act(self, policy, sample):
        x = self.inputs()
        mask = IDENTITY[self.assignment].astype(bool)
        for i in np.flatnonzero(self.ready):
            trial = self.low.trials[i]
            chefs = ordered_chefs(trial.controller)
            carrying = np.array([trial.env.state.chefs[j].held is not None for j in chefs])
            current = ASSIGNMENTS[self.assignment[i]]
            mask[i] = np.all((ASSIGNMENTS == current) | ~carrying, axis=1)
        dist = Categorical(logits=policy.allocator(tensor(x)).masked_fill(~tensor(mask),-1e9))
        selected = dist.sample() if sample else dist.logits.argmax(-1)
        ids = selected.numpy()
        self.metrics["choice_opportunities"] += int((mask.sum(1)>1).sum())
        for i in np.flatnonzero(ids != self.assignment):
            self.end_spell(i)
            self.metrics["assignment_changes"] += 1
            trial = self.low.trials[i]
            home = ordered_chefs(self.home[i])
            new = [home[j] for j in ASSIGNMENTS[ids[i]]]
            trial.controller.server,trial.controller.main,trial.controller.side = new
            self.low.left[i] = 0
        self.assignment = ids.copy()
        self.age[self.ready] = 0
        moves, lower, worker = self.low.act(policy,sample)
        return moves, (x,mask,ids,dist.log_prob(selected).numpy(),x), lower, worker

    def advance(self, moves, reset):
        chefs = [ordered_chefs(t.controller) for t in self.low.trials]
        rewards, ended, infos, _, _ = self.low.advance(moves,reset)
        self.age += 1
        self.spell += 1
        for i,info in enumerate(infos):
            perm = ASSIGNMENTS[self.assignment[i]]
            for e in info["events"]:
                if e["type"] not in WORK:
                    continue
                for role,chef in enumerate(chefs[i]):
                    if chef in e["actors"] and perm[role]!=role:
                        self.participants[i].add(chef)
                        self.metrics["reassigned_work_events"] += 1
            self.ready[i] = self.age[i]>=16 or (self.age[i]>=6 and
                                                  any(e["type"] in STAGES for e in info["events"]))
            if ended[i,0]:
                self.end_spell(i)
                if reset:
                    self.home[i]=copy(self.low.trials[i].controller)
                    self.assignment[i]=self.age[i]=0
                    self.ready[i]=True
        return rewards,ended,infos


def evaluate(policy,seeds,trace=False):
    batch=TeamBatch(seeds,np.random.default_rng(0))
    histories=[[] for _ in seeds]
    for tick in range(1,301):
        moves,_,_,_=batch.act(policy,False)
        _,_,infos=batch.advance(moves,False)
        if trace:
            for i,info in enumerate(infos):
                histories[i].append(dict(tick=tick,actions=moves[i].tolist(),goals=batch.low.goal[i].tolist(),
                                         assignment=int(batch.assignment[i]),reward=info["team_reward"],events=info["events"]))
    rows=[dict(seed=s,horizon=300,trajectory=h,orders=info["orders"],**info["metrics"])
          for s,h,info in zip(seeds,histories,infos)]
    return rows,dict(**summary(rows),assignment_metrics=dict(batch.metrics))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    def log(stage,**kw):
        row=json.dumps(dict(stage=stage,**kw))
        print(row,flush=True)
        with (args.output/"progress.jsonl").open("a") as f:
            f.write(row+"\n")
    policy=SharedPolicy()
    policy.load_state_dict(torch.load(args.checkpoint,map_location="cpu",weights_only=True))
    policy.manager=Manager(policy.manager)
    policy.allocator=mlp(INPUTS,6)
    with torch.no_grad():
        policy.allocator[-1].weight.zero_()
        policy.allocator[-1].bias.zero_()
        policy.allocator[-1].bias[0]=3.4
    policy.assignment_value=mlp(INPUTS,1)
    high=PPO(policy.allocator,policy.assignment_value,.00032,"cpu")
    low=PPO(policy.manager,policy.critic,.00022,"cpu",worker=policy.worker)
    val=list(range(2001,2013))
    holdout=list(range(84031,84055))
    _,stats=evaluate(policy,val)
    log("initial",seed=args.seed,checkpoint=portable_path(args.checkpoint),**stats)
    if stats["min_score"]!=420 or stats["mean_score"]!=420:
        raise ValueError("The initial greedy policy must score 420")
    best,best_step=420,0
    torch.save(policy.state_dict(),args.output/"best.pt")
    batch=TeamBatch(rng.integers(2**31,size=12),rng)
    for update in range(1,85):
        top,bottom,scores=[],[],[]
        batch.metrics.clear()
        for _ in range(128):
            moves,upper,lower,worker=batch.act(policy,True)
            r,done,infos=batch.advance(moves,True)
            top.append((*upper,r[:,0],done[:,0]))
            bottom.append((*lower,r,done,*worker))
            scores.extend(x["metrics"]["team_score"] for d,x in zip(done[:,0],infos) if d)
        high.update(top,batch.inputs())
        low.update(bottom,np.repeat(encode(batch.low.trials)[0][:,None,:],3,axis=1))
        log("ppo_update",update=update,environment_steps=update*1536,sampled_scores=scores,**batch.metrics)
        if update%6==0:
            _,stats=evaluate(policy,val)
            log("validation",update=update,**stats)
            if stats["mean_score"]>best:
                best,best_step=stats["mean_score"],update
                torch.save(policy.state_dict(),args.output/"best.pt")
    torch.save(policy.state_dict(),args.output/"last.pt")
    for label in ("last","best"):
        if label=="best":
            policy.load_state_dict(torch.load(args.output/"best.pt",weights_only=True))
        rows,stats=evaluate(policy,holdout,True)
        with gzip.open(args.output/f"{label}-evaluation.json.gz","wt") as f:
            json.dump(rows,f,separators=(",",":"))
        log(label,update=84 if label=="last" else best_step,**stats)

if __name__=="__main__":
    main()
