"""CPU algebra witness only: no ML model is trained."""
from __future__ import annotations
import json
import math
from pathlib import Path

ALPHA = 0.7
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def sub(a,b): return [x-y for x,y in zip(a,b)]
def norm(a): return math.sqrt(dot(a,a))
def transpose(a): return [list(x) for x in zip(*a)]
def mv(a,b): return [dot(r,b) for r in a]
def mm(a,b):
    bt=transpose(b)
    return [[dot(r,c) for c in bt] for r in a]
def solve2(a,b):
    det=a[0][0]*a[1][1]-a[0][1]*a[1][0]
    return [(a[1][1]*b[0]-a[0][1]*b[1])/det,
            (-a[1][0]*b[0]+a[0][0]*b[1])/det]
def facts(m): return [m[0]-ALPHA*m[1]*m[1],m[1]]
def jac(m): return [[1.,-2.*ALPHA*m[1],0.],[0.,1.,0.]]
def write(m,c,leak=0.):
    a,_=facts(m)
    return [a+leak*math.tanh(m[1])+ALPHA*c*c,c,0.93*m[2]+0.2*a]
def perturb(x,i,d):
    y=x[:]; y[i]+=d; return y
def fd_grad(fn,x,h=1e-6):
    return [(fn(perturb(x,i,h))-fn(perturb(x,i,-h)))/(2*h) for i in range(len(x))]
def fd_jac(fn,x,h=1e-6):
    cols=[[v/(2*h) for v in sub(fn(perturb(x,i,h)),fn(perturb(x,i,-h)))] for i in range(len(x))]
    return transpose(cols)
def run():
    start=[1.2,0.8,-0.3]; c=-0.4; target=[1.4,0.3]; js=jac(start); results=[]
    for leak in [0.,0.1,1.8]:
        end=write(start,c,leak); je=jac(end)
        covector=sub(facts(end),target); g_end=mv(transpose(je),covector)
        recovered=solve2(mm(je,transpose(je)),mv(je,g_end))
        relay=mv(transpose(js),[recovered[0],0.])
        single=dot(je[0],g_end)/dot(je[0],je[0])
        single_relay=[v*single for v in js[0]]
        def loss(x):
            err=sub(facts(write(x,c,leak)),target)
            return 0.5*dot(err,err)
        true=fd_grad(loss,start)
        transition=fd_jac(lambda x:write(x,c,leak),start)
        defect=sub(mm([je[0]],transition)[0],js[0])
        bound=norm(defect)*abs(recovered[0])
        results.append(dict(
            leak=leak,facts_start=facts(start),facts_end=facts(end),
            true_gradient_start=true,joint_fact_relay=relay,
            identity_skip_gradient=g_end,single_fact_projection_relay=single_relay,
            semantic_covector=covector,recovered_covector=recovered,
            relay_error_l2=norm(sub(true,relay)),
            identity_skip_error_l2=norm(sub(true,g_end)),
            single_fact_projection_error_l2=norm(sub(true,single_relay)),
            invariant_jacobian_defect=defect,defect_error_bound=bound,
            relay_true_gradient_dot=dot(relay,true)))
    assert results[0]["relay_error_l2"]<1e-8
    assert results[0]["single_fact_projection_error_l2"]>0.1
    assert results[0]["identity_skip_error_l2"]>0.1
    assert all(r["relay_error_l2"]<=r["defect_error_bound"]+1e-8 for r in results)
    return dict(purpose="CPU algebra and finite-difference check only; no training.",
        start_memory=start,new_b=c,target_facts=target,results=results,
        lesson="Joint fact coordinates avoid cross-fact gradient contamination. Exact local semantic invariance yields an exact bridge for its loss component; native invariance alone is insufficient.")
if __name__=="__main__":
    result=run()
    Path(__file__).with_name("semantic_adjoint_relay_algebra_result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
