"""CUDA-backed, support-constrained neural policy improvement from native short transitions."""
import argparse
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from prep_value import EXEC, Replay, physical, state_key
from v2.policy import SharedPolicy, features
from v2.train import Trial, act, encode, summary


def observation(t):
    state=t.env.state
    active=[o for o in state.orders if o.status!="served"]
    order=np.zeros((2,4),np.float32)
    for i,o in enumerate(active):
        order[i,int(o.kind=="joint")*2+int(o.status=="started")]=1
    # Use observable kitchen state, without absolute ticket IDs, score or phase clock.
    return np.concatenate((features(t.env,t.controller)[:-2],order.ravel(),
                           np.array([len(state.orders)%2],np.float32)))


def horizon_input(remaining):
    f=remaining.float()
    return torch.stack((f/300,torch.log1p(f)/np.log(301)),dim=-1)


def dataset(replay):
    device="cuda"
    n=len(replay.edges)
    successors=torch.zeros((n,216),device=device,dtype=torch.long)
    payoff=torch.full((n,216),-torch.inf,device=device)
    t=Trial(488,"multi",EXEC)
    states=[]
    support={}
    for s,es in enumerate(replay.edges):
        t.env.state=replay.snapshots[s]
        states.append(observation(t))
        codes=[]
        for (a,b,c),(dest,reward) in es.items():
            code=36*a+6*b+c
            successors[s,code]=dest
            payoff[s,code]=reward/20
            codes.append(code)
        if codes:
            support[state_key(t)]=codes
    mask=payoff.isfinite()
    values=torch.zeros(n,device=device)
    targets=[]
    for h in range(1,301):
        q=payoff+.995*values[successors]
        values=q.max(1).values.masked_fill(~mask.any(1),0.)
        # Monotone bounded transform of relative Bellman values preserves action ranking.
        targets.append(torch.expm1((q-values[:,None])*80))
    return (torch.as_tensor(np.stack(states),device=device),mask,torch.stack(targets),support)


@torch.no_grad()
def evaluate(net,base,support,seeds,restricted,trace=False):
    trials=[Trial(s,"multi",EXEC) for s in seeds]
    histories=[[] for _ in seeds]
    fallback=decisions=0
    for tick in range(300):
        obs=torch.as_tensor(np.stack([observation(t) for t in trials]),device="cuda")
        h=horizon_input(torch.full((len(seeds),),300-tick,device="cuda"))
        raw=net(torch.cat((obs,h),1)).cpu().numpy()
        infos=[]
        for i,t in enumerate(trials):
            allowed=support.get(state_key(t)) if restricted else range(216)
            if allowed:
                decisions+=int(len(allowed)>1)
                code=allowed[np.argmax(raw[i,list(allowed)])]
                m=physical(t,(code//36,(code//6)%6,code%6))
            else:
                _,ox,mk,learn=encode([t])
                m=act([t],base,ox,mk,learn,None)[0][0]
                fallback+=1
            r,_,info=t.step(m)
            if trace:
                histories[i].append(dict(tick=tick+1,actions=m.tolist(),events=info["events"],reward=r))
            infos.append(info)
    rows=[dict(seed=s,horizon=300,trajectory=h,orders=i["orders"],**i["metrics"])
          for s,h,i in zip(seeds,histories,infos)]
    return rows,dict(**summary(rows),fallback_steps=fallback,decisions_with_choices=decisions)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--proposals",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(488)
    torch.set_float32_matmul_precision("high")
    start=time.monotonic()
    def log(stage,**kw):
        msg=json.dumps(dict(stage=stage,seconds=round(time.monotonic()-start,2),**kw))
        print(msg,flush=True)
        with (a.output/"progress.jsonl").open("a") as f:
            f.write(msg+"\n")
    base=SharedPolicy()
    base.load_state_dict(torch.load(a.checkpoint,weights_only=True,map_location="cpu"))
    buf=Replay()
    prefix_scores=[buf.collect(base)]
    for plan in sorted(a.proposals.glob("plan-*.json")):
        prefix_scores.append(buf.collect(base,json.loads(plan.read_text())))
    attempts,retained=buf.local_exploration()
    x,mask,targets,support=dataset(buf)
    usable=torch.where(mask.any(1))[0]
    torch.save(support,a.output/"support.pt")
    log("data",gpu=torch.cuda.get_device_name(),states=len(x),prefix_scores=prefix_scores,
        local_steps=attempts,retained=retained,edges=int(mask.sum()),prefix_length=90)
    for width in (128,256):
        torch.manual_seed(488)
        net=nn.Sequential(nn.Linear(x.shape[1]+2,width),nn.ReLU(),nn.Linear(width,width),
                          nn.ReLU(),nn.Linear(width,216)).cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-5)
        best=-1
        for update in range(1,6001):
            ids=usable[torch.randint(len(usable),(1024,),device="cuda")]
            h=torch.randint(1,301,(1024,),device="cuda")
            prediction=net(torch.cat((x[ids],horizon_input(h)),1))
            sq=(prediction-targets[h-1,ids]).square()
            legal=mask[ids]
            loss=(sq*legal).sum()/legal.sum()+sq.mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if update in (500,1200,2400,4000,6000):
                _,vs=evaluate(net,base,support,[2011],True)
                _,vu=evaluate(net,base,support,[2011],False)
                metric=(vs["mean_score"]+vu["mean_score"]/1000)
                log("validation",width=width,update=update,loss=loss.item(),supported=vs["mean_score"],free=vu["mean_score"])
                if metric>best:
                    torch.save(net.state_dict(),a.output/f"q-{width}.pt");best=metric
                if min(vs["mean_score"],vu["mean_score"])>=440:
                    break
        net.load_state_dict(torch.load(a.output/f"q-{width}.pt",weights_only=True))
        for restricted in (True,False):
            rows,report=evaluate(net,base,support,list(range(84031,84055)),restricted,True)
            with gzip.open(a.output/f"q-{width}-{'supported' if restricted else 'free'}-evaluation.json.gz","wt") as f:
                json.dump(rows,f,separators=(",",":"))
            log("test",width=width,restricted=restricted,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
                parameters=sum(p.numel() for p in net.parameters()),**report)
if __name__=="__main__":
    main()
