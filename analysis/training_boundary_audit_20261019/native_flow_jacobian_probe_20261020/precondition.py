"""Host-only solver study on a previously computed Jacobian.

Spectral preconditioners below use the full matrix as an oracle, so this is
not a demonstrated cheap native preconditioner or a policy training result.
"""
import json
import os
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, cg

HERE=Path(__file__).resolve().parent


def main():
    j=np.load(HERE/'reverse_jacobian.npy').astype(np.float64)
    c=j@j.T
    n=len(c)
    diagonal=np.diag(c)
    preconditioners={'diagonal':lambda b:b/diagonal}
    for name,groups in [('temporal_per_coordinate',[np.arange(i,n,23) for i in range(23)]),
                        ('coordinate_per_timestep',[np.arange(t*23,(t+1)*23) for t in range(16)])]:
        factors=[cho_factor(c[np.ix_(g,g)],lower=True) for g in groups]
        def solve(b,groups=groups,factors=factors):
            result=np.zeros_like(b)
            for g,factor in zip(groups,factors):result[g]=cho_solve(factor,b[g])
            return result
        preconditioners[name]=solve
    eigenvectors,singular,_=np.linalg.svd(j,full_matrices=False)
    eigenvalues=singular**2
    for rank in [16,32,64]:
        u=eigenvectors[:,:rank]
        vals=eigenvalues[:rank]
        residual_diagonal=diagonal-np.sum(u*u*vals,axis=1)
        assert np.all(residual_diagonal>0)
        dinv=1/residual_diagonal
        middle=cho_factor(np.diag(1/vals)+u.T@(dinv[:,None]*u),lower=True)
        def solve(b,u=u,dinv=dinv,middle=middle):
            d=dinv*b
            return d-dinv*(u@cho_solve(middle,u.T@d))
        preconditioners[f'oracle_top{rank}_plus_diagonal']=solve
    rng=np.random.default_rng(20261020)
    rhs=[rng.normal(size=n) for _ in range(3)]
    results={}
    for name,inverse in preconditioners.items():
        m=LinearOperator((n,n),matvec=inverse,dtype=np.float64)
        rows=[]
        for b in rhs:
            count=[0]
            def callback(_):count[0]+=1
            x,info=cg(c,b,M=m,rtol=1e-8,atol=0,maxiter=4000,callback=callback)
            rows.append(dict(iterations=count[0],info=int(info),
                relative_residual=float(np.linalg.norm(c@x-b)/np.linalg.norm(b)),
                lift_amplification=float(np.linalg.norm(j.T@x)/np.linalg.norm(b))))
        results[name]=rows
    # SVD lift solves J w = b directly, without squaring the condition number.
    direct=[]
    for b in rhs:
        coeff=eigenvectors.T@b
        w=j.T@(eigenvectors@(coeff/(singular**2)))
        direct.append(dict(relative_residual=float(np.linalg.norm(j@w-b)/np.linalg.norm(b)),
                           lift_amplification=float(np.linalg.norm(w)/np.linalg.norm(b))))
    report=dict(preconditioned_cg=results,direct_svd_lift=direct,
        caveats=['All analyses use one finite-precision numerical Jacobian.',
                 'Oracle spectral and exact block preconditioners require the full matrix here; cost of estimating them without the matrix is not tested.',
                 'Convergent linear algebra does not demonstrate low gradient variance or policy improvement.'])
    (HERE/'preconditioner_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
