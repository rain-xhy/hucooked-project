"""Search repeating two-order schedules with fixed interaction ports in SharedKitchen."""
import argparse
from collections import deque
from functools import lru_cache
import json

from ortools.sat.python import cp_model
from overcooked_ai_py.cooperative.shared.layouts import FLOOR

# Each interaction pose is (x, y, facing_x, facing_y).
D, L, K = (1,5,-1,0), (5,1,1,0), (5,2,1,0)
M, E, W = (4,1,0,-1), (4,2,-1,0), (2,2,1,0)
A, B, S = (2,3,1,0), (4,3,-1,0), (1,3,-1,0)


@lru_cache(None)
def travel(start, finish):
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        pose, cost = queue.popleft()
        if pose == finish:
            return cost
        x, y = pose[:2]
        for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
            q = (x+dx,y+dy)
            nxt = (*(q if q in FLOOR else (x,y)),dx,dy)
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, cost+1))
    raise ValueError(finish)


def solve(period, assembly, normal_port, seconds, *, variant=0, owners=None):
    model = cp_model.CpModel()
    times, roles, offsets, jobs = {}, {}, {}, {}
    for n in range(2):
        for key, poses in dict(plate=[D,A], main=[M,E], raw=[L,K], chop=[K,K],
                               side=[K,B], assemble=[assembly[n]], serve=[A,S],
                               unload=[normal_port if n==0 else E],
                               **({"joint":[W]} if n else {})).items():
            task = (n,key)
            jobs[task] = poses
            times[task] = model.new_int_var(-period,2*period, str(task))
            phase = model.new_int_var(0,period-1,"phase")
            offset = model.new_int_var(-1,2,"offset")
            model.add(times[task] == phase + period*offset)
            offsets[task] = phase
            roles[task] = model.new_int_var(0,2,"role")
    def duration(task):
        poses = jobs[task]
        return (travel(poses[0],poses[-1])+1) if len(poses)>1 else 0
    def precede(first, second, gap=1, shift=0, end_first=True, end_second=False):
        model.add(times[second] + shift + (duration(second) if end_second else 0)
                  >= times[first] + (duration(first) if end_first else 0) + gap)
    model.add(times[0,"assemble"]==0)
    model.add(times[1,"assemble"]>=1)
    model.add(times[1,"assemble"]<period)
    model.add(roles[0,"plate"]==0)
    for n in range(2):
        nx=1-n
        shift=period if n else 0
        for x in ("plate","side","unload"):
            precede((n,x),(n,"assemble"))
        precede((n,"assemble"),(n,"serve"))
        precede((n,"main"),(n,"unload"),5)
        precede((n,"raw"),(n,"chop"))
        precede((n,"chop"),(n,"side"))
        precede((n,"side"),(nx,"raw"),shift=shift,end_first=False,end_second=True)
        precede((n,"unload"),(nx,"main"),shift=shift,end_second=True)
        precede((n,"serve"),(nx,"plate"),shift=shift,end_first=False,end_second=True)
        precede((n,"assemble"),(nx,"side"),shift=shift,end_second=True)
        precede((n,"assemble"),(nx,"unload"),shift=shift)
        precede((n,"serve"),(n,"main"),shift=period,end_second=True)
    model.add(times[1,"unload"]==times[1,"joint"])
    model.add(roles[1,"unload"]!=roles[1,"joint"])
    tasks=list(jobs)
    for i,a in enumerate(tasks):
        for b in tasks[i+1:]:
            same=model.new_bool_var("same")
            model.add(roles[a]==roles[b]).only_enforce_if(same)
            model.add(roles[a]!=roles[b]).only_enforce_if(same.Not())
            wraps=model.new_bool_var("wrap")
            forward=offsets[b]-offsets[a]+period*wraps
            model.add(forward >= duration(a)+travel(jobs[a][-1],jobs[b][0])+1).only_enforce_if(same)
            model.add(period-forward >= duration(b)+travel(jobs[b][-1],jobs[a][0])+1).only_enforce_if(same)
    if owners:
        for task, role in owners.items():
            model.add(roles[task] == role)
    if variant:
        import random
        rng = random.Random(variant)
        # Vary stage lead times to generate alternative feasible stationary flows.
        objective = sum(rng.randint(-4, 4) * times[t] for t in tasks)
        model.minimize(objective)
    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=seconds
    solver.parameters.num_search_workers=4
    status=solver.solve(model)
    result={"period":period,"assembly":assembly,"normal_port":normal_port,"status":solver.status_name(status)}
    if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        result["tasks"]=[dict(order=n,job=k,start=solver.value(times[n,k]),role=solver.value(roles[n,k]),
                              poses=jobs[n,k],duration=duration((n,k))) for n,k in jobs]
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds",type=float,default=12)
    p.add_argument("--periods", nargs="+", type=int, default=[26,27,28])
    args=p.parse_args()
    for period in args.periods:
        for normal_port in (E,W):
            for assembly in ((A,A),(B,A),(A,B),(B,B)):
                print(json.dumps(solve(period,assembly,normal_port,args.seconds)),flush=True)
