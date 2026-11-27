"""Route candidate cyclic assignments on the grid and rank them by native return."""
import argparse
from itertools import permutations
import json
from pathlib import Path

from ortools.sat.python import cp_model
from overcooked_ai_py.cooperative.shared import SharedKitchen, SharedKitchenConfig
from overcooked_ai_py.cooperative.shared.kitchen import DIRECTIONS, INTERACT, STAY
from overcooked_ai_py.cooperative.shared.layouts import FLOOR, SPAWNS

tiles = sorted(FLOOR)
poses = [(*tile,*d) for tile in tiles for d in DIRECTIONS]
pose_id = {p:i for i,p in enumerate(poses)}
motions = {}
for i, (x,y,ox,oy) in enumerate(poses):
    motions[i] = {i:STAY}
    for a,(dx,dy) in enumerate(DIRECTIONS):
        dest=(x+dx,y+dy)
        j=pose_id[(*(dest if dest in FLOOR else (x,y)),dx,dy)]
        if i!=j:
            motions[i][j]=a


def route(sched, offset, starts):
    period=sched["period"]
    repeat=offset+period
    horizon=repeat+period
    events=[{} for _ in starts]
    for cycle in range(5):
        for task in sched["tasks"]:
            t=offset+cycle*period+task["start"]
            for k,pose in enumerate(task["poses"]):
                tick=t+(task["duration"] if k else 0)
                if tick<1:
                    return
                if tick<=horizon:
                    events[task["role"]][tick]=pose_id[tuple(pose)]
    model=cp_model.CpModel()
    paths=[]
    locations=[]
    for start, required in zip(starts,events):
        initial=pose_id[(*start,0,-1)]
        reachable={initial}
        layers=[]
        for t in range(1,horizon+1):
            if t in required:
                j=required[t]
                edges=[(j,j)] if j in reachable else []
            else:
                edges=[(i,j) for i in reachable for j in motions[i]]
            if not edges:
                return
            layers.append(edges)
            reachable={j for _,j in edges}
        for t in reversed(range(horizon)):
            layers[t]=[(i,j) for i,j in layers[t] if j in reachable]
            reachable={i for i,_ in layers[t]}
        path=[model.new_int_var(0,len(poses)-1,"pose") for _ in range(horizon+1)]
        loc=[model.new_int_var(0,len(tiles)-1,"tile") for _ in path]
        for v,p in zip(loc,path):
            model.add_division_equality(v,p,4)
        for t,edges in enumerate(layers):
            model.add_allowed_assignments([path[t],path[t+1]],edges)
        model.add(path[repeat]==path[-1])
        paths.append(path)
        locations.append(loc)
    for t in range(horizon+1):
        model.add_all_different([row[t] for row in locations])
        if t:
            for i,j in ((0,1),(0,2),(1,2)):
                cross=model.new_bool_var("cross")
                model.add(locations[i][t]!=locations[j][t-1]).only_enforce_if(cross)
                model.add(locations[j][t]!=locations[i][t-1]).only_enforce_if(cross.Not())
    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=2
    solver.parameters.num_search_workers=2
    if solver.solve(model) not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        return
    sequence=[]
    for t in range(1,horizon+1):
        sequence.append([INTERACT if t in required else motions[solver.value(path[t-1])][solver.value(path[t])]
                         for path,required in zip(paths,events)])
    return dict(starts=starts, prefix=sequence[:repeat], cycle=sequence[repeat:],schedule=sched, offset=offset)


def play(plan, seed):
    env=SharedKitchen(SharedKitchenConfig())
    env.reset(seed)
    order=[next(i for i,c in enumerate(env.state.chefs) if c.position==tuple(p)) for p in plan["starts"]]
    for t in range(env.config.horizon):
        moves=plan["prefix"][t] if t<len(plan["prefix"]) else plan["cycle"][(t-len(plan["prefix"]))%len(plan["cycle"])]
        actions=[STAY]*3
        for i,a in zip(order,moves):
            actions[i]=a
        env.step(actions)
    return env.episode_metrics()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedules",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    best=0
    for line in args.schedules.read_text().splitlines():
        s=json.loads(line)
        if "tasks" not in s:
            continue
        first=min(x["start"] for x in s["tasks"])
        for offset in (4-first,8-first,12-first):
            for east in permutations(SPAWNS[1:]):
                plan=route(s,offset,(SPAWNS[0],*east))
                if plan is None:
                    continue
                stats=play(plan,2001)
                print(json.dumps(dict(period=s["period"],offset=offset,score=stats["team_score"])),flush=True)
                if stats["team_score"]>best:
                    best=stats["team_score"]
                    args.output.write_text(json.dumps(plan))
                break
            else:
                continue
            break
