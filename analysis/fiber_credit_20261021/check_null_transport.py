"""Check the null-field algebra against fully differentiated projection fields."""
import sys,json
from pathlib import Path
import numpy as np
import torch
from torch.func import jacrev
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'native_score_implicit_20261021'))
from implicit_geometry import geometry
from null_transport import reference_parts,null_parts,null_surrogate

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
torch.manual_seed(27181)


def structural(m,d):
    a=torch.randn(d,m); a[:,:d]+=3*torch.eye(d)
    c=torch.randn(d,m)*.1
    theta=torch.randn(d,requires_grad=True)*.1
    ref=theta.detach().clone(); z=torch.randn(m,requires_grad=True)
    l=torch.randn(m,d)*.3
    s=torch.eye(m).flip(0)
    def f(x,p):return (a@x)*p.exp()+c@torch.tanh(x)+p
    f0=lambda x:f(x,ref)
    j=jacrev(f0)(z).detach();b,inv,_=geometry(j)
    def cfun(x):
        jj=jacrev(f0)(x)
        bb=jj.T@torch.linalg.inv(jj@jj.T)
        return l-bb@jj@l
    exact_c=cfun(z)
    exact_div=torch.einsum('idi->d',jacrev(cfun)(z))
    exact_field=lambda x:cfun(x)@(.5*(f(s@x,theta)-f(-s@x,theta)))
    exact_s=z@exact_field(z)-torch.trace(jacrev(exact_field)(z))
    exact_g=torch.autograd.grad(exact_s,theta,retain_graph=True)[0]
    divs=[];surrogates=[]
    for e in torch.eye(m)*m**.5:
        parts=reference_parts(f0,z,j,b,inv,e)
        cc,dd=null_parts(parts,l)
        divs.append(dd)
        surrogates.append(null_surrogate(lambda x:f(x,theta),z,cc,dd,e,s))
    numerical_div=torch.stack(divs).mean(0)
    numerical_g=torch.autograd.grad(torch.stack(surrogates).mean(),theta)[0]
    result={'noise_dim':m,'output_dim':d,
            'null_projection_residual':float((j@exact_c).abs().max()),
            'divergence_error':float((exact_div-numerical_div).abs().max()),
            'gradient_error':float((exact_g-numerical_g).abs().max())}
    assert max(result[k] for k in ['null_projection_residual','divergence_error','gradient_error'])<1e-9
    return result


def calibration_example():
    # F_(mu,log_sigma,kappa)(z)=mu+exp(log_sigma)z1+kappa*z2, at (0,0,0).
    # True output-law score is [z1,z1^2-1,0]. Moore-Penrose IBP has nuisance z1*z2.
    # L=e2, S swaps z1,z2, antithetic F yields null-control score [0,z2^2-1,z1*z2].
    rng=np.random.default_rng(517)
    def scores(n):
        z=rng.normal(size=(n,2));x,y=z.T
        base=np.stack([x,x*x-1,x*y],1)
        control=np.stack([x*0,y*y-1,x*y],1)
        return z,base,control
    _,b,c=scores(4096)
    den=(c*c).mean(0)
    alpha=np.divide(-(b*c).mean(0),den,out=np.zeros(3),where=den>1e-15)
    z,b,c=scores(65536); corrected=b+c*alpha
    # Same state-only baseline and same frozen controls for both rewards.
    out=[]
    for name,reward in [('positive_action',(z[:,0]>0).astype(float)-.5),
                        ('sine_action',np.sin(z[:,0]))]:
        g0=reward[:,None]*b;g1=reward[:,None]*corrected
        out.append({'reward':name,'base_mean':g0.mean(0).tolist(),'corrected_mean':g1.mean(0).tolist(),
                    'base_total_variance':float(g0.var(0,ddof=1).sum()),
                    'corrected_total_variance':float(g1.var(0,ddof=1).sum()),
                    'nuisance_gradient_max_after_correction':float(np.abs(g1[:,2]).max())})
    assert abs(alpha[2]+1)<1e-12
    assert all(r['nuisance_gradient_max_after_correction']<1e-12 for r in out)
    return {'calibration_samples':4096,'independent_evaluation_samples':65536,
            'blockwise_coefficients':alpha.tolist(),'rewards':out,
            'scope':'Affine Gaussian illustrative case only; reward-free calibration, no VLA variance claim.'}


result={'status':'null_transport_algebra_verified','structure':[structural(3,2),structural(7,3)],
        'calibration_example':calibration_example(),'new_vla_calls':0,'new_simulator_episodes':0}
(HERE/'check_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
