"""Reward-based policy search over options extracted from a fresh 420 rollout."""
import argparse
from collections import deque
import ctypes
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from overcooked_ai_py.cooperative.shared.layouts import FLOOR, STATIONS
from v2.policy import SharedPolicy, ordered_chefs
from v2.train import Trial, act, encode
from autonomous_ppo import EXEC, DIRECTIONS, HORIZON


def extract(checkpoint,seed):
    teacher=SharedPolicy()
    teacher.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True))
    t=Trial(seed,"multi",EXEC)
    actors=ordered_chefs(t.controller)
    calls=[]; chunks=[[] for _ in actors]
    events={"load":5,"place_cutting":6,"chop":7,"take_cutting":8,"place_buffer":9,
            "take_buffer":10,"assemble":13,"take_meal":14,"serve":15,"unload":16}
    for tick in range(HORIZON):
        _,obs,mask,learn=encode([t])
        moves=act([t],teacher,obs,mask,learn,None)[0][0]
        calls.append(tuple(moves[list(actors)]))
        old=t.env.state
        _,_,info=t.step(moves)
        for e in info["events"]:
            name=e["type"]
            for i in e["actors"]:
                role=actors.index(i); c=old.chefs[i]
                xy=c.position
                target=tuple(a+b for a,b in zip(xy,c.orientation))
                if name=="take_source": op={STATIONS["M"]:1,STATIONS["L"]:2,STATIONS["D"]:3}[target]
                elif name=="place_assembly": op=11 if e["slot"]=="side" else 12
                else: op=events[name]
                if name in ("take_source","unload","assemble","take_meal") or not chunks[role]:
                    chunks[role].append(dict(role=role,start=tick,tokens=[]))
                chunks[role][-1]["tokens"].append((xy,DIRECTIONS.index(c.orientation),op))
    assert t.reward==420
    # Infer repetition from the source policy's own primitive actions; no target period supplied.
    period=next(p for p in range(1,HORIZON//4) if calls[-3*p:]==calls[-4*p:-p])
    groups=[]
    for stream in chunks:
        count=sum(200<=c["start"]<200+period for c in stream)
        groups.extend(stream[:count])
    return groups,period


def navigation(goals):
    table=np.full((len(goals),49,4),4,dtype=np.uint8)
    for k,(target,heading) in enumerate(goals):
        for xy in FLOOR:
            queue=deque([(xy,4)]);seen={xy}
            while queue:
                curr,first=queue.popleft()
                if curr==target:
                    for d in range(4):
                        table[k,7*xy[1]+xy[0],d]=(5 if d==heading else heading) if xy==target else first
                    break
                for a,(dx,dy) in enumerate(DIRECTIONS):
                    nxt=curr[0]+dx,curr[1]+dy
                    if nxt in FLOOR and nxt not in seen:
                        queue.append((nxt,a if curr==xy else first));seen.add(nxt)
    return table


class Skills:
    def __init__(self,groups,library=None):
        device="cuda" if library else "cpu"
        self.groups=groups
        available={}
        for chunk in groups:
            for xy,d,_ in chunk["tokens"]:
                station=xy[0]+DIRECTIONS[d][0],xy[1]+DIRECTIONS[d][1]
                # Other approaches are enumerated for every station by the same map rule.
                available[station]=sorted((p,h) for p in FLOOR for h,(dx,dy) in enumerate(DIRECTIONS)
                                          if (p[0]+dx,p[1]+dy)==station)
        # Pot/table allow only the two horizontal ports in the native environment.
        from overcooked_ai_py.cooperative.shared.layouts import ASSEMBLY_PORTS,COOKER_PORTS
        for name,ports in (("A",ASSEMBLY_PORTS),("G",COOKER_PORTS)):
            if STATIONS[name] in available:
                available[STATIONS[name]]=[v for v in available[STATIONS[name]] if v[0] in ports]
        self.goals=sorted(set(sum(available.values(),[])))
        self.nav_np=navigation(self.goals)
        self.nav=torch.tensor(self.nav_np,device=device)
        if library:
            self.lib=ctypes.CDLL(str(library.resolve()))
            self.fn=self.lib.option_returns
            self.fn.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]*3
            self.fn.restype=None
        pieces=[];group_ids=[];sub=[]
        for g,chunk in enumerate(groups):
            for j,(xy,d,op) in enumerate(chunk["tokens"]):
                dx,dy=DIRECTIONS[d]
                target=xy[0]+dx,xy[1]+dy
                alternatives=[(xy,d)]+[v for v in available[target] if v!=(xy,d)]
                pieces.append([self.goals.index(alternatives[i%len(alternatives)])*32+op for i in range(2)])
                group_ids.append(g);sub.append(j)
        self.pieces=torch.tensor(pieces,device=device)
        self.g=torch.tensor(group_ids,device=device)
        self.sub=torch.tensor(sub,device=device)
        self.index=torch.arange(len(self.g),device=device)

    def programs(self,owner,priority,port):
        # Priority orders each individual chef's own option queue. Clocks never trigger interactions.
        idx=self.g
        g_order=priority.argsort(dim=1,stable=True).argsort(1)
        token=self.pieces[self.index,port[:,idx]]
        actors=owner[:,idx]
        ordering=actors*100000+g_order[:,idx]*100+self.sub
        perm=ordering.argsort(1)
        sorted_tokens=token.gather(1,perm)
        lengths=torch.stack([(actors==i).sum(1) for i in range(3)],1).int()
        offsets=lengths.cumsum(1)-lengths
        slot=(offsets[:,:,None]+self.index).clamp_max(len(idx)-1)
        programs=sorted_tokens[:,None,:].expand(-1,3,-1).gather(2,slot).short().contiguous()
        return programs,lengths

    def evaluate(self,owner,priority,port,trace=False):
        prog,lens=self.programs(owner,priority,port)
        n=len(prog); result=torch.empty((n,2),device="cuda")
        moves=torch.empty((n,300,3),device="cuda",dtype=torch.int32) if trace else None
        self.fn(prog.data_ptr(),lens.data_ptr(),self.nav.data_ptr(),prog.shape[-1],n,
                result.data_ptr(),moves.data_ptr() if trace else None,torch.cuda.current_stream().cuda_stream)
        return result,moves

    def native(self,owner,priority,port,seeds):
        # Recompute choices directly from native chef positions and native interaction outcomes.
        p,l=self.programs(owner,priority,port)
        programs=[row[:n].tolist() for row,n in zip(p[0].cpu(),l[0].cpu())]
        rows=[]
        for seed in seeds:
            t=Trial(seed,"multi",EXEC); actors=ordered_chefs(t.controller)
            pc=[0,0,0];hist=[]
            for tick in range(1,HORIZON+1):
                old=t.env.state; physical=np.full(3,4)
                for role,i in enumerate(actors):
                    if programs[role]:
                        g=programs[role][pc[role]]//32
                        c=old.chefs[i];x,y=c.position
                        physical[i]=self.nav_np[g,7*y+x,DIRECTIONS.index(c.orientation)]
                r,_,info=t.step(physical)
                for event in info["events"]:
                    name=event["type"]
                    for i in event["actors"]:
                        role=actors.index(i)
                        if not programs[role]: continue
                        op=programs[role][pc[role]]%32
                        labels=("","take_source","take_source","take_source","return_source","load",
                                "place_cutting","chop","take_cutting","place_buffer","take_buffer",
                                "place_assembly","place_assembly","assemble","take_meal","serve","unload")
                        if labels[op]==name: pc[role]=(pc[role]+1)%len(programs[role])
                hist.append(dict(tick=tick,actions=physical.tolist(),events=info["events"],reward=r))
            rows.append(dict(seed=seed,horizon=HORIZON,trajectory=hist,orders=info["orders"],**info["metrics"]))
        return rows


def option_ppo(skills,role,priority,out,log,args):
    """On-policy REINFORCE/PPO over temporally extended actions sampled at reset.

    Each population member is a separate actor. Selection/evaluation uses its
    deterministic distribution mode, never an individual high-return rollout.
    """
    from torch import nn
    import torch.nn.functional as F
    I,B,G=64,512,len(role)
    start_r=F.one_hot(role,3).float()*3.1
    start_p=F.one_hot(torch.zeros_like(role),2).float()*3.1
    def learn(x):
        return nn.Parameter(x.expand(I,*x.shape).clone())
    rp,pp=learn(start_r),learn(start_p)
    mu=learn(torch.zeros(G,device="cuda"))
    opt=torch.optim.Adam([rp,pp,mu],lr=.075)
    sigma=4.
    best=-1
    for update in range(1,args.generations+1):
        with torch.no_grad():
            def draw(logits):
                u=torch.rand(I,B,G,1,device="cuda")
                return (u>logits.softmax(-1)[:,None].cumsum(-1)).sum(-1).clamp_max(logits.shape[-1]-1)
            r,p=draw(rp),draw(pp)
            key=mu[:,None]+torch.randn(I,B,G,device="cuda")
            priority_actions=priority+sigma*key
            ret,_=skills.evaluate(r.reshape(-1,G),priority_actions.reshape(-1,G),p.reshape(-1,G))
            ret=ret.reshape(I,B,2)
            objective=ret[:,:,0]+ret[:,:,1]/100000.
            # Within-batch Boltzmann utility improves credit for rare high-return rollouts.
            # Only score and discounted delivery timing enter the policy objective.
            utility=((objective-objective.max(1,keepdim=True).values)/12.).exp()
            adv=(utility-utility.mean(1,keepdim=True))/(utility.std(1,keepdim=True)+1e-5)
        def logprob():
            def categorical(x,idx):
                return x.log_softmax(-1)[:,None].expand(-1,B,-1,-1).gather(-1,idx.unsqueeze(-1)).squeeze(-1).sum(-1)
            return categorical(rp,r)+categorical(pp,p)-.5*(key-mu[:,None]).square().sum(-1)
        old=logprob().detach()
        for _ in range(3):
            ratio=(logprob()-old).clamp(-15,15).exp()
            maximize=torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean(1).sum()
            opt.zero_grad();(-maximize).backward();opt.step()
        with torch.no_grad():
            roles=rp.argmax(-1); ports=pp.argmax(-1); orders=priority+sigma*mu
            det,_=skills.evaluate(roles,orders,ports)
            champ=det[:,0].argmax();score=int(det[champ,0])
            if update%5==0 or score>best:
                log(stage="option_ppo",update=update,episodes=I*B*update,
                    sampled_max=int(ret[:,:,0].max()),mean=float(ret[:,:,0].mean()),
                    greedy_max=score,greedy_mean=float(det[:,0].mean()))
            if score>best:
                best=score
                choices=[a[champ:champ+1] for a in (roles,orders,ports)]
                test=skills.native(*choices,[args.seed])
                if test[0]["team_score"]!=score:raise ValueError("Native score mismatch")
                torch.save(dict(role_logits=rp[champ].cpu(),port_logits=pp[champ].cpu(),
                                priority_offset=mu[champ].cpu(),scale=sigma,groups=skills.groups),
                           out/"policy.pt")
                (out/"manager.json").write_text(json.dumps(dict(groups=skills.groups,
                    owner=choices[0].tolist(),priority=choices[1].tolist(),port=choices[2].tolist())))
                if score>=440:
                    rows=skills.native(*choices,range(84031,84055))
                    with gzip.open(out/"official-evaluation.json.gz","wt") as f:json.dump(rows,f)
                    log(stage="native_test",update=update,scores=[v["team_score"] for v in rows])
                    if min(v["team_score"] for v in rows)>=440:return


def main(args):
    torch.manual_seed(args.seed);torch.set_num_threads(1)
    groups,period=extract(args.checkpoint,args.seed)
    skills=Skills(groups,args.library)
    out=args.output;out.mkdir(parents=True)
    def log(**msg):
        print(json.dumps(msg),flush=True)
        with (out/"progress.jsonl").open("a") as f: f.write(json.dumps(msg)+"\n")
    role=torch.tensor([g["role"] for g in groups],device="cuda")
    priority=torch.tensor([g["start"] for g in groups],device="cuda",dtype=torch.float)
    port=torch.zeros_like(role)
    base=(role[None],priority[None],port[None])
    result,_=skills.evaluate(*base)
    native=skills.native(*base,[args.seed])[0]["team_score"]
    if result[0,0].item()!=native: raise ValueError("Initial macro policy disagrees with native")
    log(stage="options_from_420",source_period=period,groups=groups,macro_start=native)
    if args.method=="ppo":
        return option_ppo(skills,role,priority,out,log,args)
    # Cross-entropy policy search with separate populations to retain alternative return plateaus.
    # Fit the categorical/continuous parameter distributions to sampled elites after each generation.
    islands,pop=48,512
    G=len(role); P=pop*islands
    onehot=torch.nn.functional.one_hot
    roles=.92*onehot(role,3).float()+.08/3
    ports=.93*onehot(port,2).float()+.07/2
    rp=roles.expand(islands,-1,-1).clone(); pp=ports.expand(islands,-1,-1).clone()
    mean=priority.expand(islands,-1).clone()
    sigma=torch.full_like(mean,4.)
    peak=-1
    for gen in range(args.generations):
        def sample(probs):
            u=torch.rand(islands,pop,G,1,device="cuda")
            return (u>probs[:,None].cumsum(-1)).sum(-1).clamp_max(probs.shape[-1]-1)
        r=sample(rp);p=sample(pp)
        w=mean[:,None]+sigma[:,None]*torch.randn(islands,pop,G,device="cuda")
        ret,_=skills.evaluate(r.reshape(P,G),w.reshape(P,G),p.reshape(P,G))
        ret=ret.reshape(islands,pop,2)
        # All fitness terms come from the timing and count of actual deliveries.
        fit=ret[:,:,0]+ret[:,:,1]/100000.
        if fit.max().item()>peak:
            peak=float(fit.max());idx=int(fit.argmax());i,j=divmod(idx,pop)
            selected=[x[i,j][None] for x in (r,w,p)]
            official=skills.native(*selected,[args.seed])[0]["team_score"]
            if official!=ret[i,j,0].item():raise ValueError((official,ret[i,j].tolist()))
            log(stage="record",generation=gen,score=official,return_with_tiebreak=peak)
            with (out/"manager.json").open("w") as f:
                json.dump(dict(groups=groups,owner=selected[0].tolist(),priority=selected[1].tolist(),
                               port=selected[2].tolist()),f)
            if official>=440:
                rows=skills.native(*selected,range(84031,84055))
                with gzip.open(out/"official-evaluation.json.gz","wt") as f:json.dump(rows,f)
                log(stage="native_test",scores=[x["team_score"] for x in rows])
                if min(x["team_score"] for x in rows)>=440: break
        elite=fit.topk(12,dim=1).indices[:,:,None].expand(-1,-1,G)
        er,ew,ep=[x.gather(1,elite) for x in (r,w,p)]
        rp=.35*rp+.65*(onehot(er,3).float().mean(1)*.90+.10/3)
        pp=.35*pp+.65*(onehot(ep,2).float().mean(1)*.90+.10/2)
        mean=.35*mean+.65*ew.mean(1)
        sigma=(.35*sigma+.65*ew.std(1)).clamp(1.5,16)
        if gen%10==0: log(stage="generation",generation=gen,episodes=(gen+1)*P,
                            sample_max=int(ret[:,:,0].max()),sample_mean=float(ret[:,:,0].mean()))
        if gen%15==14:
            # Restart weaker islands from diverse mutations of the best online elite.
            weak=fit.max(1).values.argsort()[:islands//2]
            strong=fit.max(1).values.argsort()[-islands//2:]
            rp[weak]=.65*rp[strong]+.35/3
            pp[weak]=.65*pp[strong]+.35/2
            mean[weak]=mean[strong];sigma[weak]=7.


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",type=Path)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--method",choices=("ppo","cem"),default="ppo")
    p.add_argument("--seed",type=int,default=413)
    p.add_argument("--generations",type=int,default=100)
    p.add_argument("--evaluate-policy",type=Path,
                   help="Reload an option-PPO checkpoint and evaluate directly in native SharedKitchen.")
    args=p.parse_args()
    if args.evaluate_policy:
        model=torch.load(args.evaluate_policy,map_location="cpu",weights_only=True)
        groups=model["groups"]
        skills=Skills(groups)
        owners=model["role_logits"].argmax(-1)[None]
        priorities=(torch.tensor([g["start"] for g in groups])+model["scale"]*model["priority_offset"])[None]
        ports=model["port_logits"].argmax(-1)[None]
        rows=skills.native(owners,priorities,ports,range(84031,84055))
        args.output.mkdir(parents=True,exist_ok=False)
        with gzip.open(args.output/"official-evaluation.json.gz","wt") as f: json.dump(rows,f)
        (args.output/"shared.trajectory.json").write_text(json.dumps(rows[0]))
        print(json.dumps(dict(scores=[row["team_score"] for row in rows])))
    else:
        if args.library is None or args.checkpoint is None:
            p.error("Training requires --library and --checkpoint")
        main(args)
