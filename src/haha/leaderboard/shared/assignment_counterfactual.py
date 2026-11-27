"""Paired rollouts from states reached by the original 420-point policy."""
import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from shared.team_assignment_ppo import ASSIGNMENTS, INPUTS, Manager, TeamBatch
from v2.policy import SharedPolicy, ordered_chefs
from haha.goal_training import mlp


@torch.no_grad()
def prefer(policy, first, fallback):
    head=policy.allocator[-1]
    head.bias.fill_(-100)
    head.bias[fallback]=50
    head.bias[first]=100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="420-point BC checkpoint")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(488)
    policy=SharedPolicy()
    policy.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    policy.manager=Manager(policy.manager)
    policy.allocator=mlp(INPUTS,6)
    with torch.no_grad():
        policy.allocator[-1].weight.zero_()
    prefer(policy,0,0)
    base=TeamBatch([17031],np.random.default_rng(0))
    cases=[]
    last=-31
    for tick in range(300):
        if tick<=240 and tick-last>=30 and base.ready[0]:
            trial=base.low.trials[0]
            carrying=np.array([trial.env.state.chefs[j].held is not None for j in ordered_chefs(trial.controller)])
            allowed=np.all((ASSIGNMENTS==ASSIGNMENTS[0]) | ~carrying,axis=1)
            if allowed.sum()>1:
                cases.append((tick,deepcopy(base),np.flatnonzero(allowed)[1:]))
                last=tick
        moves,*_=base.act(policy,False)
        base.advance(moves,False)
    baseline=base.low.trials[0].env.episode_metrics()
    assert baseline["team_score"]==420
    results=[]
    for tick,start,choices in cases:
        for choice in choices:
            b=deepcopy(start)
            prefer(policy,int(choice),0)
            for t in range(tick,300):
                moves,*_=b.act(policy,False)
                if t==tick:
                    assert b.assignment[0]==choice
                    prefer(policy,0,int(choice))
                b.advance(moves,False)
            stats=b.low.trials[0].env.episode_metrics()
            results.append(dict(tick=tick,assignment=ASSIGNMENTS[choice].tolist(),
                                score=stats["team_score"],delta=stats["team_score"]-baseline["team_score"],
                                movement_conflicts=stats["movement_conflicts"],duration=b.metrics["spell_ticks"],
                                reciprocal=bool(b.metrics["two_chefs_worked"])))
    print(json.dumps(dict(seed=17031,baseline=baseline,results=results)))
if __name__=="__main__":
    main()
