"""Improve a 420-point policy using native returns, optionally replaying a better searched trajectory."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.functional import cross_entropy

from haha.goal_training import tensor
from v2.policy import (FEATURE_DIM, GOALS, SharedPolicy, ordered_chefs, reserve, reserved_features,
                       valid_actions, worker_inputs)
from v2.train import Execution, Trial, encode, summary
from v2.pure_rl import PPO, widen_manager

EXECUTION = Execution(profile="continuous", goal_lock=True)
N = len(GOALS)


class ResidualActor(nn.Module):
    def __init__(self, source, width, timing):
        super().__init__()
        self.base = deepcopy(source) if width == 128 else widen_manager(source,width)
        self.inputs = source[0].in_features
        self.outputs = source[-1].out_features
        self.timing = None
        if timing:
            self.timing = nn.Sequential(nn.Linear(34+self.inputs,64),nn.Tanh(),nn.Linear(64,self.outputs))
            nn.init.zeros_(self.timing[-1].weight)
            nn.init.zeros_(self.timing[-1].bias)

    def forward(self,x):
        obs,offset,clock = x.split([self.inputs,self.outputs,34],dim=-1)
        logits = self.base(obs) + offset
        if self.timing is not None:
            logits = logits + self.timing(torch.cat([obs,clock],dim=-1))
        return logits


class Team(nn.Module):
    def __init__(self, source, width, timing):
        super().__init__()
        self.source=source.requires_grad_(False)
        self.manager=ResidualActor(source.manager,width,timing).requires_grad_(True)
        self.worker=ResidualActor(source.worker,width,timing).requires_grad_(True)
        self.critic=deepcopy(source.critic).requires_grad_(True)

    @torch.no_grad()
    def pack(self,source,obs,mask,ticks):
        raw=source(tensor(obs))
        prior=raw.masked_fill(~tensor(mask),-1e9).log_softmax(-1).clamp_min(-9)
        prior=prior.masked_fill(~tensor(mask),-5)
        angles=2*np.pi*np.asarray(ticks)[...,None]/np.arange(20,37)
        clock=np.concatenate((np.cos(angles),np.sin(angles)),axis=-1).astype(np.float32)
        return np.concatenate((obs,(prior-raw).numpy(),clock),axis=-1)

    @torch.no_grad()
    def choose(self,trials,sample=False,targets=None):
        common,obs,masks,_=encode(trials)
        ticks=np.asarray([t.env.state.tick for t in trials])
        x=self.pack(self.source.manager,obs,masks,np.broadcast_to(ticks[:,None],masks.shape[:2]))
        gd=Categorical(logits=self.manager(tensor(x)))
        g=gd.sample() if sample else gd.logits.argmax(-1)
        if targets is not None:
            g=tensor(targets[0])
        reserves=[set() for _ in trials]
        moves=np.empty((len(trials),3),int)
        ws=[]; acts=[]; probs=[]
        for r in range(3):
            ids=[ordered_chefs(t.controller)[r] for t in trials]
            goals=g[:,r].numpy()
            wx=worker_inputs(obs[:,r],goals,np.stack([reserved_features(s) for s in reserves]))
            mask=np.stack([valid_actions(t.env,i,k,goal_lock=True) for t,i,k in zip(trials,ids,goals)])
            w=self.pack(self.source.worker,wx,mask,ticks)
            md=Categorical(logits=self.worker(tensor(w)))
            a=md.sample() if sample else md.logits.argmax(-1)
            if targets is not None:
                a=tensor(targets[1][:,r])
            for j,(trial,i,move) in enumerate(zip(trials,ids,a.tolist())):
                moves[j,i]=move
                reserve(trial.env,i,move,reserves[j])
            ws.append(w); acts.append(a.numpy()); probs.append(md.log_prob(a).numpy())
        a=np.stack(acts,axis=1)
        w=np.stack(ws,axis=1)
        record=(x,np.ones((*g.shape,N),bool),g.numpy(),gd.log_prob(g).numpy()+np.stack(probs,axis=1),
                np.repeat(common[:,None,:],3,axis=1))
        return moves,record,(w,np.ones((*a.shape,6),bool),a)


def evaluate(team,seeds,trace=False):
    trials=[Trial(int(s),"multi",EXECUTION) for s in seeds]
    histories=[[] for _ in trials]
    for tick in range(1,301):
        moves,data,_=team.choose(trials)
        infos=[]
        for i,(trial,move) in enumerate(zip(trials,moves)):
            reward,_,info=trial.step(move)
            infos.append(info)
            if trace:
                histories[i].append(dict(tick=tick,actions=move.tolist(),goals=data[2][i].tolist(),
                                         events=info["events"],reward=reward))
    return [dict(seed=int(s),horizon=300,trajectory=h,orders=info["orders"],**info["metrics"])
            for s,h,info in zip(seeds,histories,infos)]


def successful_replay(team,plan,seed):
    trial=Trial(seed,"multi",EXECUTION)
    home=ordered_chefs(trial.controller)
    cols=[plan["starts"].index(list(trial.env.state.chefs[i].position)) for i in home]
    all_moves=np.array((plan["prefix"]+plan["cycle"]*300)[:300])[:,cols]
    goals=np.full_like(all_moves,-1)
    for tick,actions in enumerate(all_moves):
        for role,chef in enumerate(home):
            if actions[role]==5:
                pos=trial.env.state.chefs[chef].position
                goals[tick,role]=next(i for i,(tile,station) in enumerate(GOALS) if tile==pos and station is not None)
        physical=np.zeros(3,int)
        physical[list(home)]=actions
        trial.step(physical)
    score=trial.env.episode_metrics()["team_score"]
    if score<=420:
        return score,None
    for tick in reversed(range(299)):
        goals[tick]=np.where(goals[tick]<0,goals[tick+1],goals[tick])
    # Terminal movement with no subsequent station is not labelled.
    valid=goals>=0
    goals=np.maximum(goals,0)
    trial=Trial(seed,"multi",EXECUTION)
    xs=[]; ws=[]
    for g,a in zip(goals,all_moves):
        moves,rec,worker=team.choose([trial],targets=(g[None],a[None]))
        xs.append(rec[0][0]); ws.append(worker[0][0])
        trial.step(moves[0])
    return score,[(np.stack(xs)[valid],goals[valid]),(np.stack(ws)[valid],all_moves[valid])]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--candidate",type=Path)
    p.add_argument("--width",type=int,choices=(128,256),default=128)
    p.add_argument("--timing",action="store_true")
    p.add_argument("--seed",type=int,default=488)
    args=p.parse_args()
    out=args.output
    out.mkdir(parents=True)
    torch.manual_seed(args.seed); torch.set_num_threads(1)
    rng=np.random.default_rng(args.seed)
    def log(stage,**kw):
        line=json.dumps(dict(stage=stage,**kw))
        print(line,flush=True)
        with (out/"progress.jsonl").open("a") as f:
            f.write(line+"\n")
    source=SharedPolicy()
    source.load_state_dict(torch.load(args.checkpoint,weights_only=True,map_location="cpu"))
    team=Team(source,args.width,args.timing)
    val=list(range(2001,2013))
    initial=summary(evaluate(team,val))
    log("initial",**initial,width=args.width,timing=args.timing,seed=args.seed,
        actor_parameters=sum(p.numel() for net in (team.manager,team.worker) for p in net.parameters()))
    if initial["min_score"]!=420 or initial["mean_score"]!=420:
        raise ValueError("420 start not preserved")
    best=420
    selected="initial"
    torch.save(team.state_dict(),out/"best.pt")
    if args.candidate:
        score,data=successful_replay(team,json.loads(args.candidate.read_text()),int(rng.integers(2**31)))
        log("exploration_candidate",score=score)
        if data:
            for net,(inputs,labels) in zip((team.manager,team.worker),data):
                opt=torch.optim.Adam(net.parameters(),lr=.002)
                x,y=tensor(inputs),tensor(labels)
                for epoch in range(1200):
                    logits=net(x); loss=cross_entropy(logits,y)
                    opt.zero_grad(); loss.backward(); opt.step()
                    if loss.item()<.001:
                        break
                log("self_imitation_fit",network="manager" if net is team.manager else "worker",
                    epochs=epoch+1,label_accuracy=float((logits.argmax(-1)==y).float().mean()))
            stats=summary(evaluate(team,val))
            log("self_imitation",**stats)
            torch.save(team.state_dict(),out/"replay-fit.pt")
            if stats["mean_score"]>best:
                selected="self_imitation"; best=stats["mean_score"]
                torch.save(team.state_dict(),out/"best.pt")
    opt=PPO(team.manager,team.critic,.0002,"cpu",worker=team.worker)
    trials=[Trial(int(s),"multi",EXECUTION) for s in rng.integers(2**31,size=12)]
    for update in range(1,25):
        records=[]; scores=[]
        for _ in range(128):
            moves,rec,worker=team.choose(trials,True)
            rewards=[]; done=[]
            for i,(trial,move) in enumerate(zip(trials,moves)):
                r,d,info=trial.step(move)
                rewards.append([r/20]*3); done.append([d]*3)
                if d:
                    scores.append(info["metrics"]["team_score"])
                    trials[i]=Trial(int(rng.integers(2**31)),"multi",EXECUTION)
            records.append((*rec,np.array(rewards,np.float32),np.array(done),*worker))
        opt.update(records,np.repeat(encode(trials)[0][:,None,:],3,axis=1))
        log("ppo_update",update=update,environment_steps=update*1536,sampled_scores=scores)
        if update%4==0:
            stats=summary(evaluate(team,val)); log("validation",update=update,**stats)
            if stats["mean_score"]>best:
                best=stats["mean_score"]; selected=f"ppo-{update}"
                torch.save(team.state_dict(),out/"best.pt")
    torch.save(team.state_dict(),out/"last.pt")
    for label in ("last","best"):
        if label=="best":
            team.load_state_dict(torch.load(out/"best.pt",weights_only=True))
        rows=evaluate(team,range(84031,84055),True)
        with gzip.open(out/f"{label}-evaluation.json.gz","wt") as f:
            json.dump(rows,f,separators=(",",":"))
        log(label,selected=selected if label=="best" else "ppo-24",**summary(rows))

if __name__=="__main__":
    main()
