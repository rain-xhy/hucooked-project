"""Fit a finite-horizon joint-action value controller from short native kitchen transitions."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
from itertools import permutations, product
import json
from pathlib import Path
import pickle

import numpy as np
import torch
from periodic_schedule import A,B,E,W,solve,travel
from route_schedule import SPAWNS,route
from v2.policy import SharedPolicy, ordered_chefs
from v2.train import Execution, Trial, act, encode, summary

EXEC=Execution(profile="continuous",goal_lock=True)
PREFIX=90


def proposal(spec):
    period,assembly,port,variant=spec
    schedule=solve(period,assembly,port,1.5,variant=variant)
    if "tasks" in schedule:
        for east in permutations(SPAWNS[1:]):
            starts=(SPAWNS[0],*east)
            offset=max(1+travel((*starts[t["role"]],0,-1),tuple(t["poses"][0]))-t["start"] for t in schedule["tasks"])
            for slack in (0,2,5):
                plan=route(schedule,offset+slack,starts)
                if plan:
                    return spec,plan
    return spec,None


def state_key(trial):
    s=trial.env.state
    tickets=[o for o in s.orders if o.status!="served"]
    slots={o.id:i for i,o in enumerate(tickets)}
    def item(v):
        return None if v is None else (v.kind,slots.get(v.order_id))
    chefs=tuple((s.chefs[i].position,s.chefs[i].orientation,item(s.chefs[i].held))
                for i in ordered_chefs(trial.controller))
    return (chefs,item(s.cooker.item),s.cooker.remaining,item(s.cutting.item),s.cutting.progress,
            item(s.assembly.main),item(s.assembly.side),item(s.assembly.plate),item(s.buffer),
            tuple((o.kind,o.status) for o in tickets),len(s.orders)%2)


def physical(trial,action):
    moves=np.zeros(3,dtype=int)
    moves[list(ordered_chefs(trial.controller))]=action
    return moves


class Replay:
    def __init__(self):
        self.ids={}
        self.edges=[]
        self.snapshots=[]

    def identify(self,trial):
        key=state_key(trial)
        if key not in self.ids:
            self.ids[key]=len(self.edges)
            self.edges.append({})
            self.snapshots.append(trial.env.state)
        return self.ids[key]

    def add(self,source,action,target,reward):
        edge=(target,int(reward))
        if self.edges[source].setdefault(tuple(action),edge)!=edge:
            raise ValueError("Conflicting native transition for the same encoded state and action")

    def collect(self,policy,plan=None):
        trial=Trial(488,"multi",EXEC)
        if plan:
            sequence=plan["prefix"]+plan["cycle"]*PREFIX
            starts=list(map(tuple,plan["starts"]))
            cols=[starts.index(trial.env.state.chefs[i].position)
                  for i in ordered_chefs(trial.controller)]
        for t in range(PREFIX):
            before=self.identify(trial)
            if plan:
                action=[sequence[t][j] for j in cols]
            else:
                _,obs,masks,learn=encode([trial])
                moves,_,_=act([trial],policy,obs,masks,learn,None)
                action=moves[0][list(ordered_chefs(trial.controller))]
            reward,_,_=trial.step(physical(trial,action))
            self.add(before,action,self.identify(trial),reward)
        return trial.reward

    def local_exploration(self):
        trial=Trial(488,"multi",EXEC)
        attempts=retained=0
        for source,(snapshot,edges) in enumerate(zip(self.snapshots,self.edges)):
            choices=set()
            for a in edges:
                for role,move in product(range(3),range(6)):
                    b=list(a); b[role]=move; choices.add(tuple(b))
            for action in sorted(choices-edges.keys()):
                trial.env.state=snapshot
                reward=trial.env.step(physical(trial,action))[1]
                attempts+=1
                target=self.ids.get(state_key(trial))
                if target is not None:
                    self.add(source,action,target,reward)
                    retained+=1
        return attempts,retained

    def fit(self,zero_reward=False):
        n=len(self.edges)
        # A state without outgoing data has no estimated future delivery credits.
        width=max(map(len,self.edges))
        next_ids=np.zeros((n,width),dtype=int)
        rewards=np.full((n,width),-np.inf)
        actions=np.zeros((n,width,3),dtype=np.int8)
        for i,edges in enumerate(self.edges):
            for j,(a,(target,reward)) in enumerate(edges.items()):
                actions[i,j]=a
                next_ids[i,j]=target
                rewards[i,j]=0 if zero_reward else reward
        values=np.zeros(n)
        policy=np.zeros((301,n,3),dtype=np.int8)
        for remaining in range(1,301):
            q=rewards+.9999*values[next_ids]
            idx=q.argmax(1)
            values=q[np.arange(n),idx]
            values[~np.isfinite(values)]=0
            policy[remaining]=actions[np.arange(n),idx]
        return policy


def evaluate(replay,learned,baseline,seeds):
    rows=[]; fallback=0
    for seed in seeds:
        trial=Trial(seed,"multi",EXEC)
        history=[]
        for t in range(300):
            i=replay.ids.get(state_key(trial))
            if i is None or not replay.edges[i]:
                _,obs,masks,learning=encode([trial])
                moves=act([trial],baseline,obs,masks,learning,None)[0][0]
                fallback+=1
            else:
                moves=physical(trial,learned[300-t,i])
            reward,_,info=trial.step(moves)
            history.append(dict(tick=t+1,actions=list(map(int,moves)),reward=reward,events=info["events"]))
        rows.append(dict(seed=seed,horizon=300,trajectory=history,orders=info["orders"],**info["metrics"]))
    return rows,dict(**summary(rows),fallback_steps=fallback)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--variants",type=int,default=4)
    args=p.parse_args()
    args.out.mkdir(parents=True)
    torch.set_num_threads(1)
    baseline=SharedPolicy()
    baseline.load_state_dict(torch.load(args.checkpoint,weights_only=True,map_location="cpu"))
    replay=Replay()
    replay.collect(baseline)
    test=range(84031,84055)
    def measure(name,zero=False):
        policy=replay.fit(zero)
        rows,report=evaluate(replay,policy,baseline,test)
        print(json.dumps(dict(stage=name,states=len(replay.edges),edges=sum(map(len,replay.edges)),**report)),flush=True)
        with gzip.open(args.out/f"{name}-evaluation.json.gz","wt") as f:
            json.dump(rows,f,separators=(",",":"))
        if name=="local-q":
            with gzip.open(args.out/"controller.pkl.gz","wb") as f:
                pickle.dump((replay.ids,policy,[bool(e) for e in replay.edges]),f)
    measure("baseline-q")
    budgets=product((24,25,26,27,28),product((A,B),repeat=2),(E,W),range(args.variants))
    with ProcessPoolExecutor(max_workers=3) as pool, (args.out/"prefixes.jsonl").open("w") as f:
        for spec,plan in pool.map(proposal,budgets):
            score=replay.collect(baseline,plan) if plan else None
            f.write(json.dumps(dict(spec=spec,steps=PREFIX if plan else 0,score=score))+"\n"); f.flush()
    measure("prefix-q")
    attempts,retained=replay.local_exploration()
    print(json.dumps(dict(stage="native_branches",steps=attempts,retained=retained)),flush=True)
    measure("reward-ablated",True)
    measure("local-q")

if __name__=="__main__":
    main()
