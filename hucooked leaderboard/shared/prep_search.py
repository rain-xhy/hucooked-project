"""Find physically executable two-ticket plans without stored plans or demonstrations."""
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations, product
import json
from pathlib import Path
import sys

from periodic_schedule import A,B,E,W,solve,travel
from route_schedule import SPAWNS,play,route


def generate(spec):
    period,ports,pot,variant=spec
    schedule=solve(period,ports,pot,1.2,variant=variant)
    if "tasks" not in schedule:
        return spec,schedule["status"],None,None
    best,chosen=None,None
    for east in permutations(SPAWNS[1:]):
        starts=(SPAWNS[0],*east)
        first=max(1+travel((*starts[t["role"]],0,-1),tuple(t["poses"][0]))-t["start"] for t in schedule["tasks"])
        for delay in (0,2,5):
            plan=route(schedule,first+delay,starts)
            if plan:
                score=play(plan,2001)["team_score"]
                if best is None or score>best:
                    best,chosen=score,plan
                break
    return spec,schedule["status"],best,chosen


def main():
    folder=Path(sys.argv[1])
    folder.mkdir()
    best=0
    rounds=int(sys.argv[2])
    with ProcessPoolExecutor(max_workers=3) as pool, (folder/"results.jsonl").open("w") as output:
        for v in range(rounds):
            specs=product((25,26,27,24,28),product((A,B),repeat=2),(E,W),(v,))
            for spec,status,score,plan in pool.map(generate,specs):
                print(json.dumps(dict(spec=spec,status=status,score=score)),file=output,flush=True)
                if plan is not None:
                    key=f"plan-{len(list(folder.glob('plan-*.json'))):04d}.json"
                    (folder/key).write_text(json.dumps(plan))
                    if score>best:
                        best=score
                        (folder/"candidate.json").write_text(json.dumps(plan))
            if best>=440:
                break

if __name__=="__main__":
    main()
