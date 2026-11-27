"""Clone a collision-free cyclic teacher, then train both hierarchy levels with native team return."""
import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.functional import cross_entropy

from haha.goal_training import mlp, tensor
from v2.policy import FEATURE_DIM, IDENTITY, features, ordered_chefs
from v2.pure_rl import PPO
from v2.train import CONTINUOUS_ENVIRONMENT_COMMIT, Execution, Trial, summary

JOBS = ("plate", "main", "raw", "chop", "side", "assemble", "serve", "unload", "joint")


class Team(nn.Module):
    def __init__(self, prefix, period):
        super().__init__()
        self.prefix, self.period = prefix, period
        self.clock = np.eye(prefix + period, dtype=np.float32)
        self.labels = np.eye(len(JOBS), dtype=np.float32)
        width = FEATURE_DIM + 3 + len(self.clock)
        self.manager = mlp(width,len(JOBS))
        self.worker = mlp(width + len(JOBS),6)
        self.critic = mlp(width,1)

    def inputs(self,trials):
        rows=[]
        for trial in trials:
            t=trial.env.state.tick
            phase=t if t<self.prefix else self.prefix+(t-self.prefix)%self.period
            state=np.concatenate((features(trial.env,trial.controller),self.clock[phase]))
            rows.append(np.concatenate((np.broadcast_to(state,(3,len(state))),IDENTITY),axis=1))
        return np.stack(rows)

    def motions(self,obs,goals):
        return np.concatenate((obs,self.labels[goals]),axis=-1)

    @torch.no_grad()
    def act(self,obs,sample):
        manager=Categorical(logits=self.manager(tensor(obs)))
        goal=manager.sample() if sample else manager.logits.argmax(-1)
        wx=self.motions(obs,goal.numpy())
        worker=Categorical(logits=self.worker(tensor(wx)))
        action=worker.sample() if sample else worker.logits.argmax(-1)
        return goal.numpy(),wx,action.numpy(),(manager.log_prob(goal)+worker.log_prob(action)).numpy()


def teacher_table(plan):
    # Label each approach, interaction and departure with the next scheduled job.
    schedule=plan["schedule"]
    table=[[] for _ in range(3)]
    for cycle in range(300//schedule["period"]+4):
        for task in schedule["tasks"]:
            end=plan["offset"]+cycle*schedule["period"]+task["start"]+task["duration"]
            table[task["role"]].append((end,JOBS.index(task["job"])))
    table=[sorted(row) for row in table]
    labels=np.array([[next(job for end,job in row if end>=tick) for row in table] for tick in range(1,301)])
    seq=plan["prefix"]
    actions=np.array((seq+plan["cycle"]*301)[:300])
    return labels,actions


def new(seed):
    return Trial(int(seed),"multi",Execution(profile="continuous"))


def step(trial,actions):
    physical=np.empty(3,dtype=int)
    physical[list(ordered_chefs(trial.controller))]=actions
    reward,done,info=trial.step(physical)
    return reward,done,info,physical.tolist()


def evaluate(team,seeds,teacher=None,trace=False):
    trials=[new(seed) for seed in seeds]
    histories=[[] for _ in trials]
    for t in range(300):
        if teacher is None:
            goals,_,moves,_=team.act(team.inputs(trials),False)
        else:
            goals,moves=([row[t]]*len(trials) for row in teacher)
        for i,trial in enumerate(trials):
            reward,_,info,physical=step(trial,moves[i])
            if trace:
                histories[i].append(dict(tick=t+1,actions=physical,goals=goals[i].tolist(),reward=reward,events=info["events"]))
    return [dict(seed=int(seed),horizon=300,orders=trial.env._info([],0)["orders"],trajectory=history,
                 **trial.env.episode_metrics()) for trial,seed,history in zip(trials,seeds,histories)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    plan=json.loads(args.plan.read_text())
    out=args.output
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(488)
    rng=np.random.default_rng(488)
    started=monotonic()
    def log(stage,**kw):
        line=json.dumps(dict(stage=stage,seconds=round(monotonic()-started,2),**kw))
        print(line,flush=True)
        with (out/"progress.jsonl").open("a") as f:
            f.write(line+"\n")
    team=Team(len(plan["prefix"]),len(plan["cycle"]))
    teacher=teacher_table(plan)
    # Use fixed spawn order for policy inputs regardless of solver role numbering.
    from overcooked_ai_py.cooperative.shared.layouts import SPAWNS
    ids=[plan["starts"].index(list(pos)) for pos in SPAWNS]
    goals,actions=(x[:,ids] for x in teacher)
    test=range(84031,84055)
    log("config",revision=CONTINUOUS_ENVIRONMENT_COMMIT,prefix=team.prefix,period=team.period,
        role_order=SPAWNS,seed=488,mask=False,train_episodes_bc=1,phase_observed=True)
    expert=evaluate(team,test,(goals,actions))
    (out/"teacher-evaluation.json").write_text(json.dumps(expert))
    log("teacher",**summary(expert))
    trial=new(488)
    obs=[]
    for move in actions:
        obs.append(team.inputs([trial])[0])
        step(trial,move)
    ox=np.stack(obs)
    wx=team.motions(ox,goals)
    for model,inputs,labels in ((team.manager,ox,goals),(team.worker,wx,actions)):
        x=tensor(inputs.reshape(-1,inputs.shape[-1]))
        y=tensor(labels.ravel())
        optim=torch.optim.Adam(model.parameters(),lr=.003)
        for epoch in range(1200):
            logits=model(x)
            loss=cross_entropy(logits,y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if loss.item()<.0008:
                break
        log("bc_fit",network="manager" if model is team.manager else "worker",
            epochs=epoch+1,accuracy=float((logits.argmax(-1)==y).float().mean()))
    val=range(2001,2013)
    best=summary(evaluate(team,val))["mean_score"]
    bc=evaluate(team,test,trace=True)
    (out/"bc-evaluation.json").write_text(json.dumps(bc))
    log("after_imitation",environment_steps=0,**summary(bc))
    torch.save(team.state_dict(),out/"bc.pt")
    torch.save(team.state_dict(),out/"best.pt")
    best_update=0
    learner=PPO(team.manager,team.critic,.00020,"cpu",worker=team.worker)
    lanes=12
    trials=[new(seed) for seed in rng.integers(1000000,size=lanes)]
    for update in range(1,13):
        records=[]
        for _ in range(192):
            x=team.inputs(trials)
            g,w,a,lp=team.act(x,True)
            reward,done=[],[]
            for i,trial in enumerate(trials):
                r,d,_,_=step(trial,a[i])
                reward.append([r/20]*3)
                done.append([d]*3)
                if d:
                    trials[i]=new(rng.integers(2**31))
            records.append((x,np.ones((*g.shape,len(JOBS)),bool),g,lp,x,np.array(reward,np.float32),
                            np.array(done),w,np.ones((*a.shape,6),bool),a))
        learner.update(records,team.inputs(trials))
        stats=summary(evaluate(team,val))
        log("validation",update=update,environment_steps=update*lanes*192,**stats)
        if stats["mean_score"]>best:
            best,best_update=stats["mean_score"],update
            torch.save(team.state_dict(),out/"best.pt")
    torch.save(team.state_dict(),out/"last.pt")
    final=evaluate(team,test,trace=True)
    (out/"last-evaluation.json").write_text(json.dumps(final))
    log("last-evaluation",update=12,environment_steps=27648,**summary(final))
    team.load_state_dict(torch.load(out/"best.pt",weights_only=True))
    rows=evaluate(team,test,trace=True) if best_update else bc
    (out/"best-evaluation.json").write_text(json.dumps(rows))
    log("best-evaluation",update=best_update,**summary(rows))

if __name__=="__main__":
    main()
