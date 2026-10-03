#!/usr/bin/env python3
"""Conic synthesis, structural adjoints and certified proximal density updates.

This module writes numerical data and plots. It never writes manuscript text.
All coordinates, patch dimensions and tolerances are recorded in the output.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import finite_element as base
import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
from scipy.optimize import minimize

ct = base.ct
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIG = ROOT / "figures"


def sym(a):
    return (a + a.T) / 2


class PhysicalPatchFE(base.CantileverFE):
    """Fixed area-average force and displacement/velocity measurement patch."""

    def _patch_force(self):
        out = np.zeros(self.ndof)
        # The same rectangle on every mesh. Every mesh used here resolves it.
        for ex in range(self.cfg.nelx):
            for ey in range(self.cfg.nely):
                x, y = (ex + .5) * self.dx, (ey + .5) * self.dy
                if x >= .54 - 1e-12 and .05 - 1e-12 <= y <= .15 + 1e-12:
                    for ix, iy in [(ex, ey), (ex+1, ey), (ex+1, ey+1), (ex, ey+1)]:
                        out[2*self.node(ix, iy)+1] -= self.dx*self.dy/4
        return out / np.sum(np.abs(out))

    def _nondesign_elements(self):
        return np.array([ex*self.cfg.nely+ey for ex in range(self.cfg.nelx)
                         for ey in range(self.cfg.nely)
                         if (ex+.5)*self.dx <= .06 + 1e-12
                         or ((ex+.5)*self.dx >= .54-1e-12
                             and .05-1e-12 <= (ey+.5)*self.dy <= .15+1e-12)], dtype=int)

    def _density_filter(self):
        centers = np.array([[(ex+.5)*self.dx/.06, (ey+.5)*self.dy/.05]
                            for ex in range(self.cfg.nelx) for ey in range(self.cfg.nely)])
        distances = la.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        filt = np.maximum(0, 1.55-distances)
        return filt / filt.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class Settings:
    mu: float = 1e-6
    brl_margin: float = 1e-8
    coupling_margin: float = 1e-4
    small_gain_margin: float = .10
    gap_normalized: float = .0005
    solver_tol: float = 1e-10
    max_iterations: int = 24
    projected_tolerance: float = 2e-3
    initial_proximal: float = 50.
    move_limit: float = .12


SET = Settings()
SOLVER_OPTIONS = {"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": SET.solver_tol,
                  "MSK_DPAR_INTPNT_CO_TOL_PFEAS": SET.solver_tol,
                  "MSK_DPAR_INTPNT_CO_TOL_DFEAS": SET.solver_tol,
                  "MSK_IPAR_NUM_THREADS": 1}


def lmi_numpy(p, x, gamma):
    A,B1,B2,C1,C2,D11,D12,D21 = p
    X,Y,Ah,Bh,Ch,Dh = x
    aa=A@X+B2@Ch
    bb=Y@A+Bh@C2
    ab=Ah.T+A+B2@Dh@C2
    ac=B1+B2@Dh@D21
    ad=(C1@X+D12@Ch).T
    bc=Y@B1+Bh@D21
    bd=(C1+D12@Dh@C2).T
    cd=(D11+D12@Dh@D21).T
    return np.block([[aa+aa.T,ab,ac,ad], [ab.T,bb+bb.T,bc,bd],
                     [ac.T,bc.T,-gamma*np.eye(B1.shape[1]),cd],
                     [ad.T,bd.T,cd.T,-gamma*np.eye(C1.shape[0])]])


def lmi_seeds(p, x, Z):
    """Exact pullback to A, B1, B2, C1 and C2 with Frobenius pairing."""
    A,B1,B2,C1,C2,D11,D12,D21=p
    X,Y,Ah,Bh,Ch,Dh=x
    n,nd,ne=len(A),B1.shape[1],C1.shape[0]
    i,j,k,l=slice(0,n),slice(n,2*n),slice(2*n,2*n+nd),slice(2*n+nd,None)
    z11,z12,z13,z14=Z[i,i],Z[i,j],Z[i,k],Z[i,l]
    z22,z23,z24=Z[j,j],Z[j,k],Z[j,l]
    a=2*z11@X+2*z12+2*Y@z22
    b1=2*z13+2*Y@z23
    b2=2*z11@Ch.T+2*z12@C2.T@Dh.T+2*z13@D21.T@Dh.T
    c1=2*z14.T@X+2*z24.T
    c2=2*Dh.T@B2.T@z12+2*Bh.T@z22+2*Dh.T@D12.T@z24.T
    return a,b1,b2,c1,c2


def dual_coefficients(p, Z, S, gamma_multiplier):
    A,B1,B2,C1,C2,D11,D12,D21=p
    n,nd=len(A),B1.shape[1]
    i,j,k,l=slice(0,n),slice(n,2*n),slice(2*n,2*n+nd),slice(2*n+nd,None)
    z11,z12,z13,z14=Z[i,i],Z[i,j],Z[i,k],Z[i,l]
    z22,z23,z24,z34=Z[j,j],Z[j,k],Z[j,l],Z[k,l]
    gx=sym(A.T@z11+z11@A+2*z14@C1-S[i,i])
    gy=sym(A@z22+z22@A.T+2*z23@B1.T-S[j,j])
    ga=2*z12.T
    gb=2*z22@C2.T+2*z23@D21.T
    gc=2*B2.T@z11+2*D12.T@z14.T
    gd=2*(B2.T@z12@C2.T+B2.T@z13@D21.T+D12.T@z24.T@C2.T+D12.T@z34.T@D21.T)
    gg=1-np.trace(Z[k,k])-np.trace(Z[l,l])-gamma_multiplier
    return [gx,gy,ga,gb,gc,gd,np.array(gg)]


def positive(a):
    eig,u=la.eigh(sym(a))
    return (u*np.maximum(eig,0))@u.T


def recover(p, x):
    A,B1,B2,C1,C2,D11,D12,D21=p
    X,Y,Ah,Bh,Ch,Dh=x
    n=len(A)
    # A balanced SVD factorization avoids a poorly scaled realization gauge.
    u,s,vh=la.svd(np.eye(n)-Y@X)
    Nc=u*np.sqrt(s)
    Mc=vh.T*np.sqrt(s)
    Dk=Dh
    Ck=la.solve(Mc,(Ch-Dk@C2@X).T).T
    Bk=la.solve(Nc,Bh-Y@B2@Dk)
    temp=Ah-Y@A@X-Y@B2@Dk@C2@X-Nc@Bk@C2@X-Y@B2@Ck@Mc.T
    Ak=la.solve(Mc,la.solve(Nc,temp).T).T
    controller=ct.ss(Ak,Bk,Ck,Dk)
    Pi1=np.block([[X,np.eye(n)],[Mc.T,np.zeros((n,n))]])
    Pi2=np.block([[np.eye(n),Y],[np.zeros((n,n)),Nc.T]])
    P=sym(la.solve(Pi1.T,Pi2.T).T)
    return controller,P


def closed_loop(p, ctrl):
    A,B1,B2,C1,C2,D11,D12,D21=p
    Ak,Bk,Ck,Dk=ctrl.A,ctrl.B,ctrl.C,ctrl.D
    aa=np.block([[A+B2@Dk@C2,B2@Ck],[Bk@C2,Ak]])
    bb=np.vstack([B1+B2@Dk@D21,Bk@D21])
    cc=np.hstack([C1+D12@Dk@C2,D12@Ck])
    dd=D11+D12@Dk@D21
    return ct.ss(aa,bb,cc,dd)


class ConicSynthesis:
    def __init__(self, n, settings=SET):
        self.n,self.settings=n,settings
        self.params=[cp.Parameter(shape) for shape in [(n,n),(n,4),(n,1),(3,n),(1,n),(3,4),(3,1),(1,4)]]
        A,B1,B2,C1,C2,D11,D12,D21=self.params
        self.x=[cp.Variable((n,n),symmetric=True),cp.Variable((n,n),symmetric=True),
                cp.Variable((n,n)),cp.Variable((n,1)),cp.Variable((1,n)),cp.Variable((1,1))]
        X,Y,Ah,Bh,Ch,Dh=self.x
        self.gamma=cp.Variable()
        aa=A@X+B2@Ch
        bb=Y@A+Bh@C2
        ab=Ah.T+A+B2@Dh@C2
        ac=B1+B2@Dh@D21
        ad=(C1@X+D12@Ch).T
        bc=Y@B1+Bh@D21
        bd=(C1+D12@Dh@C2).T
        cd=(D11+D12@Dh@D21).T
        self.L=cp.bmat([[aa+aa.T,ab,ac,ad],[ab.T,bb+bb.T,bc,bd],
                        [ac.T,bc.T,-self.gamma*np.eye(4),cd],
                        [ad.T,bd.T,cd.T,-self.gamma*np.eye(3)]])
        self.Q=cp.bmat([[X,np.eye(n)],[np.eye(n),Y]])
        self.reg=sum(cp.sum_squares(v) for v in self.x+[self.gamma])/2
        self.constraints=[self.Q >> settings.coupling_margin*np.eye(2*n),
                          self.L << -settings.brl_margin*np.eye(2*n+7),self.gamma>=0]
        self.problem=cp.Problem(cp.Minimize(self.gamma+settings.mu*self.reg),self.constraints)

    def solve(self, plant):
        start=time.perf_counter()
        for p,v in zip(self.params,plant): p.value=v
        self.problem.solve(solver=cp.MOSEK,warm_start=True,ignore_dpp=True,mosek_params=SOLVER_OPTIONS)
        if self.problem.status not in (cp.OPTIMAL,cp.OPTIMAL_INACCURATE):
            raise RuntimeError('synthesis status '+self.problem.status)
        x=[v.value.copy() for v in self.x]
        gamma=float(self.gamma.value)
        eta=float(self.problem.value)
        Z=positive(self.constraints[1].dual_value)
        S=positive(self.constraints[0].dual_value)
        z0=max(0.,float(self.constraints[2].dual_value))
        coeff=dual_coefficients(plant,Z,S,z0)
        zero=[np.zeros_like(v) for v in x]
        L0=lmi_numpy(plant,zero,0)+self.settings.brl_margin*np.eye(2*self.n+7)
        Q0=np.block([[-self.settings.coupling_margin*np.eye(self.n),np.eye(self.n)],
                     [np.eye(self.n),-self.settings.coupling_margin*np.eye(self.n)]])
        dual=float(np.sum(Z*L0)-np.sum(S*Q0)-sum(np.sum(c*c) for c in coeff)/(2*self.settings.mu))
        stationarity=[c+self.settings.mu*v for c,v in zip(coeff,x+[np.array(gamma)])]
        qslack=np.block([[x[0],np.eye(self.n)],[np.eye(self.n),x[1]]])-self.settings.coupling_margin*np.eye(2*self.n)
        lslack=-lmi_numpy(plant,x,gamma)-self.settings.brl_margin*np.eye(2*self.n+7)
        comp=[]
        for slack,mult in [(qslack,S),(lslack,Z)]:
            scale_s=max(1,la.norm(slack,2))
            scale_z=max(1,la.norm(mult,2))
            comp.append({'minimum_sum_eigenvalue':float(la.eigvalsh(slack/scale_s+mult/scale_z)[0]),
                         'relative_complementarity':float(la.norm(slack@mult)/max(1,la.norm(slack)*la.norm(mult))),
                         'minimum_slack_eigenvalue':float(la.eigvalsh(slack)[0])})
        ctrl,P=recover(plant,x)
        cl=closed_loop(plant,ctrl)
        a,b,c,d=cl.A,cl.B,cl.C,cl.D
        brl=np.block([[a.T@P+P@a,P@b,c.T],[b.T@P,-gamma*np.eye(4),d.T],[c,d,-gamma*np.eye(3)]])
        info={'objective':eta,'gamma':gamma,'regularized_dual':dual,'regularized_gap':eta-dual,
              'relative_kkt_residual':float(np.sqrt(sum(np.sum(v*v) for v in stationarity))/max(1,np.sqrt(sum(np.sum(v*v) for v in coeff)))),
              'coupling_slack':comp[0],'brl_slack':comp[1],
              'physical_brl_maximum_eigenvalue':float(la.eigvalsh(sym(brl))[-1]),
              'minimum_storage_eigenvalue':float(la.eigvalsh(P)[0]),
              'pole_abscissa':float(np.max(la.eigvals(cl.A).real)),
              'sdp_seconds':time.perf_counter()-start}
        return {'x':x,'gamma':gamma,'eta':eta,'Z':Z,'S':S,'controller':ctrl,'storage':P,'info':info}

    def lower_bound(self, plant):
        for p,v in zip(self.params,plant): p.value=v
        # The unrestricted optimum is an infimum. No controller is recovered here.
        cons=[self.Q>>0,self.L<<0,self.gamma>=0]
        prob=cp.Problem(cp.Minimize(self.gamma),cons)
        prob.solve(solver=cp.MOSEK,ignore_dpp=True,mosek_params=SOLVER_OPTIONS)
        Z,S=positive(cons[1].dual_value),positive(cons[0].dual_value)
        x=[v.value.copy() for v in self.x]
        zero=[np.zeros_like(v) for v in x]
        Q0=np.block([[np.zeros((self.n,self.n)),np.eye(self.n)],[np.eye(self.n),np.zeros((self.n,self.n))]])
        value=float(np.sum(Z*lmi_numpy(plant,zero,0))-np.sum(S*Q0))
        coeff=dual_coefficients(plant,Z,S,max(0,float(cons[2].dual_value)))
        return {'primal':float(prob.value),'dual':value,'gap':float(prob.value-value),
                'dual_stationarity_residual':float(np.sqrt(sum(np.sum(c*c) for c in coeff))),
                '_Z':Z,'_S':S,'_x':x}


def rayleigh_weights(lam, alpha, beta):
    damp=alpha+beta*lam
    under=damp*damp<2*lam
    q=1/lam
    qp=-1/lam**2
    root=np.sqrt(lam[under]-damp[under]**2/4)
    q[under]=1/(damp[under]*root)
    qp[under]=-q[under]*(beta/damp[under]+(.5)*(1-beta*damp[under]/2)/(lam[under]-damp[under]**2/4))
    return q,qp,1/damp,-beta/damp**2


def divided_difference(lam, f, fp):
    dl=lam[:,None]-lam[None,:]
    df=f[:,None]-f[None,:]
    close=np.abs(dl)<=1e-9*np.maximum(1,np.maximum(lam[:,None],lam[None,:]))
    out=np.divide(df,dl,out=np.zeros_like(dl),where=~close)
    out[close]=np.broadcast_to((fp[:,None]+fp[None,:])/2,out.shape)[close]
    return out


class CertifiedStudy:
    def __init__(self,cfg=base.CFG,scales=None,settings=SET,state_scale=None):
        self.cfg,self.settings=cfg,settings
        self.fe=PhysicalPatchFE(cfg)
        self.old=base.Study(self.fe,cfg)
        self.scales=scales or self.old.scales
        self.old.scales=self.scales
        self.m=cfg.retained_modes
        self.synthesis=ConicSynthesis(2*self.m,settings)
        self.state_scale=state_scale
        self.volume_gradient=(np.ones(self.fe.ne)@self.physical_jacobian())/self.fe.ne
        xlo=np.full(len(self.fe.design),cfg.rho_min)
        self.volume_offset=self.fe.volume(xlo)-self.volume_gradient@xlo
        if state_scale is None:
            mdl=self.modal(self.old.x_uniform)
            self.state_scale=1.
            p=self.plant(mdl)
            self.state_scale=float(np.sqrt(la.norm(np.vstack([p[3],p[4]]))/la.norm(p[2])))

    def physical_jacobian(self):
        J=self.fe.filter_matrix[:,self.fe.design].copy()
        J[self.fe.nondesign,:]=0
        return J

    def modal(self,x,anchor=None):
        start=time.perf_counter()
        k,m,xp=self.fe.assemble(x)
        lam,phi=la.eigh(k,m,check_finite=False)
        R=np.eye(self.m)
        if anchor is not None:
            cross=phi[:,:self.m].T@anchor['M']@anchor['phi']
            u,s,vh=la.svd(cross)
            R=u@vh
        ret=phi[:,:self.m]@R
        lr=R.T@np.diag(lam[:self.m])@R
        return {'K':k,'M':m,'lam':lam,'all_phi':phi,'phi':ret,'Lambda':lr,'rho':xp,
                'gap':float((lam[self.m]-lam[self.m-1])/self.scales.omega_scale**2),
                'eigen_seconds':time.perf_counter()-start}

    def plant(self,mdl,full=False):
        phi=mdl['all_phi'] if full else mdl['phi']
        L=np.diag(mdl['lam']) if full else mdl['Lambda']
        nm=phi.shape[1]
        w=self.scales.omega_scale
        b=phi.T@self.fe.force
        st=self.state_scale
        a=np.block([[np.zeros((nm,nm)),np.eye(nm)],[-L/w**2,-(self.scales.alpha*np.eye(nm)+self.scales.beta*L)/w]])
        b2=np.r_[np.zeros(nm),b*st*self.cfg.force_reference/(w*self.scales.vref)][:,None]
        c1=np.vstack([np.r_[b/st,np.zeros(nm)],np.zeros(2*nm),np.zeros(2*nm)])
        c2=np.r_[np.zeros(nm),b/st][None,:]
        b1=np.column_stack([b2[:,0],np.zeros((2*nm,3))])
        d11=np.array([[0,0,1,0],[0,0,0,0],[1,0,0,0]],float)
        d12=np.array([[0],[self.cfg.control_weight],[1]],float)
        d21=np.array([[0,1,0,1]],float)
        return a,b1,b2,c1,c2,d11,d12,d21

    def tail(self,mdl,gradient=False):
        lam=mdl['lam']
        phi=mdl['all_phi']
        b=phi.T@self.fe.force
        alpha,beta=self.scales.alpha,self.scales.beta
        q,qp,v,vp=rayleigh_weights(lam,alpha,beta)
        q[:self.m]=0; qp[:self.m]=0
        v[:self.m]=0; vp[:self.m]=0
        aq=self.cfg.force_reference/self.scales.qref
        av=self.cfg.force_reference/self.scales.vref
        dq=aq*np.sum(b*b*q)
        if '_c_solution' not in mdl:
            solve_start=time.perf_counter()
            damping=sparse.csc_matrix(alpha*mdl['M']+beta*mdl['K'])
            mdl['_c_solution']=spla.spsolve(damping,self.fe.force)
            mdl['_c_seconds']=time.perf_counter()-solve_start
        velocity_trace=float(self.fe.force@mdl['_c_solution']-np.sum(b[:self.m]**2/(alpha+beta*lam[:self.m])))
        dv=av*velocity_trace
        delta=float(np.hypot(dq,dv))
        zeta=(alpha+beta*lam[self.m:])/(2*np.sqrt(lam[self.m:]))
        zmin=np.min(zeta)
        kappa=1/(2*zmin*np.sqrt(1-zmin*zmin)) if zmin<1/np.sqrt(2) else 1.
        oldq=aq*kappa*np.sum(b[self.m:]**2/lam[self.m:])
        oldv=av/(2*zmin)*np.sum(b[self.m:]**2/np.sqrt(lam[self.m:]))
        out={'delta':delta,'delta_q':float(dq),'delta_v':float(dv),
             'floor_delta':float(np.hypot(oldq,oldv)),
             'mass_port_squared':float(np.sum(b*b))}
        if gradient:
            f=(dq*aq*q+dv*av*v)/delta
            fp=(dq*aq*qp+dv*av*vp)/delta
            F=divided_difference(lam,f,fp)
            G=divided_difference(lam,lam*f,f+lam*fp)
            seed=np.outer(b,b)
            out['Kseed']=sym(phi@(F*seed)@phi.T)
            out['Mseed']=-sym(phi@(G*seed)@phi.T)
        return out

    def structural_gradient(self,mdl,syn,certified=True):
        start=time.perf_counter()
        p=self.plant(mdl)
        a,b1,b2,c1,c2=lmi_seeds(p,syn['x'],syn['Z'])
        m=self.m; w=self.scales.omega_scale; st=self.state_scale
        barL=sym(-a[m:,:m]/w**2-self.scales.beta*a[m:,m:]/w)
        # Both force columns and both physical output channels contribute.
        barb=(b1[m:,0]+b2[m:,0])*st*self.cfg.force_reference/(w*self.scales.vref)+(c1[0,:m]+c2[0,m:])/st
        phi=mdl['phi']; lr=mdl['Lambda']; force=self.fe.force[:,None]
        br=phi.T@force
        W=force@barb[None,:]
        tail_start=time.perf_counter()
        tail=self.tail(mdl)
        eta=syn['eta']
        margin=1-tail['delta']*eta
        directK=np.zeros_like(mdl['K']); directM=np.zeros_like(mdl['M'])
        if certified:
            barL/=margin**2
            W/=margin**2
            vcoef=eta**2*tail['delta_v']*self.cfg.force_reference/(tail['delta']*self.scales.vref*margin**2)
            R=la.inv(self.scales.alpha*np.eye(m)+self.scales.beta*lr)
            barL+=self.scales.beta*vcoef*R@br@br.T@R
            W-=2*vcoef*force@br.T@R
            invseed=vcoef*np.outer(mdl['_c_solution'],mdl['_c_solution'])
            directK=-self.scales.beta*invseed
            directM=-self.scales.alpha*invseed
        tail_seconds=time.perf_counter()-tail_start
        theta=sym(lr@barL)+.5*sym(phi.T@W)
        deflated_start=time.perf_counter()
        # Exactly m deflated structural systems, independent of density count.
        lamb,U=la.eigh(lr)
        phid=phi@U
        Wd=W@U
        Xi=np.zeros_like(phi)
        mphi=mdl['M']@phid
        for i in range(m):
            shift=sparse.csc_matrix(mdl['K']-lamb[i]*mdl['M'])
            saddle=sparse.bmat([[shift,sparse.csc_matrix(mphi)],
                                [sparse.csc_matrix(mphi.T),sparse.csc_matrix((m,m))]],format='csc')
            rhs=np.r_[Wd[:,i]-mphi@(phid.T@Wd[:,i]),np.zeros(m)]
            Xi[:,i]=spla.spsolve(saddle,rhs)[:len(phi)]
        Xi=Xi@U.T
        deflated_seconds=time.perf_counter()-deflated_start
        Kseed=sym(phi@barL@phi.T-Xi@phi.T)+directK
        Mseed=sym(-phi@theta@phi.T+Xi@lr@phi.T)+directM
        tail_start=time.perf_counter()
        if certified:
            q,qp,_,_=rayleigh_weights(mdl['lam'],self.scales.alpha,self.scales.beta)
            q[:m]=0; qp[:m]=0
            ap=mdl['all_phi']; ab=ap.T@self.fe.force
            qcoef=eta**2*tail['delta_q']*self.cfg.force_reference/(tail['delta']*self.scales.qref*margin**2)
            rankseed=np.outer(ab,ab)
            F=divided_difference(mdl['lam'],q,qp)
            G=divided_difference(mdl['lam'],mdl['lam']*q,q+mdl['lam']*qp)
            Kseed+=qcoef*sym(ap@(F*rankseed)@ap.T)
            Mseed-=qcoef*sym(ap@(G*rankseed)@ap.T)
        tail_seconds+=time.perf_counter()-tail_start
        start=time.perf_counter()
        ke_seed=np.zeros(self.fe.ne)
        me_seed=np.zeros(self.fe.ne)
        global_k=np.zeros((self.fe.ndof,self.fe.ndof))
        global_m=np.zeros_like(global_k)
        free=np.ix_(self.fe.free_dofs,self.fe.free_dofs)
        global_k[free],global_m[free]=Kseed,Mseed
        for e,ed in enumerate(self.fe.edofs):
            ke_seed[e]=np.sum(global_k[np.ix_(ed,ed)]*self.fe.ke_unit)*self.cfg.young*(1-self.cfg.e_floor)*self.cfg.simp_p*mdl['rho'][e]**(self.cfg.simp_p-1)
            me_seed[e]=np.sum(global_m[np.ix_(ed,ed)]*self.fe.me_unit)*self.cfg.material_density
        grad=self.physical_jacobian().T@(ke_seed+me_seed)
        return grad,{'deflated_seconds':deflated_seconds,'tail_gradient_seconds':tail_seconds,
                     'damping_solve_seconds':mdl['_c_seconds'],
                     'contraction_seconds':time.perf_counter()-start}

    def evaluate(self,x,gradient=False,certified=True,anchor=None):
        mdl=self.modal(x,anchor)
        syn=self.synthesis.solve(self.plant(mdl))
        tail=self.tail(mdl)
        margin=1-tail['delta']*syn['eta']
        value=syn['eta']/margin if certified else syn['eta']
        result={'x':np.asarray(x).copy(),'modal':mdl,'synthesis':syn,'tail':tail,
                'margin':margin,'value':float(value)}
        if gradient:
            result['gradient'],result['timing']=self.structural_gradient(mdl,syn,certified)
        return result

    def project(self,target):
        lo,hi=self.cfg.rho_min,1.
        weights=self.volume_gradient
        offset=self.volume_offset
        left,right=-1e5,1e5
        for _ in range(80):
            t=(left+right)/2
            x=np.clip(target-t*weights,lo,hi)
            if offset+weights@x>self.cfg.volume_fraction: left=t
            else: right=t
        return np.clip(target-((left+right)/2)*weights,lo,hi)

    def optimize(self,x0,certified=True,name='certified'):
        accepted=self.evaluate(x0,gradient=True,certified=certified)
        history=[]
        accepted_path=[]
        proximal=self.settings.initial_proximal
        previous=None
        for iteration in range(self.settings.max_iterations+1):
            x,g=accepted['x'],accepted['gradient']
            pg=float(la.norm(x-self.project(x-g)))
            row={'iteration':iteration,'objective':accepted['value'],'eta':accepted['synthesis']['eta'],
                 'gamma':accepted['synthesis']['gamma'],'delta':accepted['tail']['delta'],
                 'small_gain_margin':accepted['margin'],'gap':accepted['modal']['gap'],
                 'projected_residual':pg,**accepted.get('timing',{}),
                 'proximal_parameter':proximal,
                 'sdp_seconds':accepted['synthesis']['info']['sdp_seconds'],
                 'eigen_seconds':accepted['modal']['eigen_seconds'],
                 'physical_brl_max_eigenvalue':accepted['synthesis']['info']['physical_brl_maximum_eigenvalue'],
                 'regularized_gap':accepted['synthesis']['info']['regularized_gap'],
                 'kkt_residual':accepted['synthesis']['info']['relative_kkt_residual']}
            history.append(row)
            accepted_path.append(x.copy())
            print(name,iteration,'objective',row['objective'],'projected',pg,flush=True)
            np.savez(DATA/f'{name}_checkpoint.npz',x=x,x_path=np.asarray(accepted_path),history=np.array(history,dtype=object))
            if pg<=self.settings.projected_tolerance or iteration==self.settings.max_iterations: break
            if previous is not None:
                sx=x-previous['x']; yg=g-previous['gradient']
                if sx@yg>1e-12:
                    proximal=max(.2,min(1e4,(sx@yg)/(sx@sx)))
            previous=accepted
            for attempt in range(18):
                target=self.project(x-g/proximal)
                step=target-x
                max_move=np.max(np.abs(step))
                if max_move>self.settings.move_limit:
                    proximal*=max_move/self.settings.move_limit
                    continue
                trial=self.evaluate(target,certified=certified,anchor=accepted['modal'])
                admissible=(trial['margin']>=self.settings.small_gain_margin
                            and trial['modal']['gap']>=self.settings.gap_normalized
                            and trial['synthesis']['info']['pole_abscissa']<-1e-8
                            and trial['synthesis']['info']['physical_brl_maximum_eigenvalue']<0)
                if admissible and trial['value']<=accepted['value']-.1*proximal*(step@step)+1e-7:
                    trial['gradient'],trial['timing']=self.structural_gradient(trial['modal'],trial['synthesis'],certified)
                    accepted=trial
                    break
                proximal*=2
            else:
                raise RuntimeError('proximal line search failed')
        return accepted,pd.DataFrame(history)


def hn(system):
    value,peak=ct.linfnorm(system,tol=1e-9)
    return float(value),float(peak)


def performance_plant(p):
    A,B1,B2,C1,C2,D11,D12,D21=p
    return A,B1[:,:2],B2,C1[:2],C2,D11[:2,:2],D12[:2],D21[:,:2]


def frequency_values(study,mdl,ctrl,omega,retained=False):
    m=study.m if retained else len(mdl['lam'])
    lam=mdl['lam'][:m]
    b=mdl['all_phi'][:,:m].T@study.fe.force
    ss=1j*np.asarray(omega)
    denom=lam[:,None]+(study.scales.alpha+study.scales.beta*lam[:,None])*ss[None,:]+ss[None,:]**2
    mech=np.sum(b[:,None]**2/denom,axis=0)
    q=study.cfg.force_reference/study.scales.qref*mech
    v=study.cfg.force_reference/study.scales.vref*ss*mech
    kval=np.array([np.asarray(ct.evalfr(ctrl,z/study.scales.omega_scale)).item() for z in ss])
    closed=1/(1-kval*v)
    mat=np.zeros((len(ss),2,2),dtype=complex)
    mat[:,0,0]=q*closed
    mat[:,0,1]=q*kval*closed
    mat[:,1,0]=study.cfg.control_weight*kval*v*closed
    mat[:,1,1]=study.cfg.control_weight*kval*closed
    return np.linalg.svd(mat,compute_uv=False)[:,0]


def regularity_rank(p,syn,settings=SET):
    """Kernel-compressed constraint Jacobian, with an explicit rank tolerance."""
    n=len(p[0]); x=syn['x']
    Q=np.block([[x[0],np.eye(n)],[np.eye(n),x[1]]])-settings.coupling_margin*np.eye(2*n)
    S=-lmi_numpy(p,x,syn['gamma'])-settings.brl_margin*np.eye(2*n+7)
    rows=[]
    dims=[]
    iu=np.triu_indices(n)
    def pack(parts):
        packed=[]
        for k,a in enumerate(parts):
            if k<2:
                z=a[iu].copy()
                z[iu[0]!=iu[1]]*=np.sqrt(2)
                packed.append(z)
            else: packed.append(np.ravel(a))
        return np.concatenate(packed)
    for matrix,which in [(Q,'coupling'),(S,'brl')]:
        eig,U=la.eigh(sym(matrix))
        # Absolute threshold in the documented dimensionless coordinates.
        V=U[:,eig<2e-6]
        dims.append(V.shape[1])
        for i in range(V.shape[1]):
            for j in range(i,V.shape[1]):
                seed=np.outer(V[:,i],V[:,j])
                if i!=j: seed=(seed+seed.T)/np.sqrt(2)
                if which=='coupling':
                    parts=dual_coefficients(p,np.zeros_like(S),seed,1.)
                else: parts=dual_coefficients(p,seed,np.zeros_like(Q),1.)
                rows.append(pack(parts))
    J=np.asarray(rows)
    values=la.svdvals(J) if len(rows) else np.array([1.])
    threshold=1e-9*values[0]
    return {'kernel_dimensions':dims,'rows':len(rows),'columns':len(pack(dual_coefficients(p,np.zeros_like(S),np.zeros_like(Q),1.))),
            'jacobian_rank':int(np.sum(values>threshold)),
            'minimum_singular_value':float(values[-1]),'maximum_singular_value':float(values[0]),
            'relative_rank_threshold':1e-9,'kernel_eigenvalue_threshold':2e-6}


def report(study,results):
    rows=[]; quality=[]; archive={}; metadata={}
    for name,r in results.items():
        mdl,syn,tail=r['modal'],r['synthesis'],r['tail']
        p=study.plant(mdl)
        aug=closed_loop(p,syn['controller'])
        full=closed_loop(performance_plant(study.plant(mdl,True)),syn['controller'])
        full_abscissa=float(ct.poles(full).real.max())
        assert full_abscissa<0 and r['margin']>0
        achieved,_=hn(aug)
        full_norm,peak=hn(full)
        reduced,_=hn(closed_loop(performance_plant(p),syn['controller']))
        nm=len(mdl['lam'])
        idx=np.r_[np.arange(study.m,nm),np.arange(nm+study.m,2*nm)]
        pf=study.plant(mdl,True)
        tail_system=ct.ss(pf[0][np.ix_(idx,idx)],pf[2][idx,:],np.vstack([pf[3][0,idx],pf[4][0,idx]]),np.zeros((2,1)))
        tail_norm,_=hn(tail_system)
        zw,_=hn(aug[:2,:2]); zp,_=hn(aug[:2,3]); rw,_=hn(aug[2,:2]); up,_=hn(aug[2,3])
        feedback_margin=1-tail['delta_v']*up
        block=zw+(zp*tail['delta_v']+tail['delta_q'])*rw/feedback_margin
        lower=study.synthesis.lower_bound(p)
        rank=regularity_rank(p,syn,study.settings)
        q=syn['info']
        row={'design':name,'eta_mu':syn['eta'],'gamma_mu':syn['gamma'],'achieved_eta':achieved,
             'eta_lb':lower['dual'],'synthesis_slack':syn['eta']-lower['dual'],
             'delta':tail['delta'],'delta_q':tail['delta_q'],'delta_v':tail['delta_v'],
             'tail_norm':tail_norm,'floor_delta':tail['floor_delta'],
             'rayleigh_to_tail_ratio':tail['delta']/tail_norm,'floor_to_tail_ratio':tail['floor_delta']/tail_norm,
             'small_gain_margin':r['margin'],'Gamma_mu':syn['eta']/r['margin'],'Gamma_block':block,
             'feedback_margin':feedback_margin,'full_norm':full_norm,'retained_norm':reduced,
             'full_peak_rad_s':peak*study.scales.omega_scale,'gap_normalized':mdl['gap'],
             'volume':study.fe.volume(r['x']),'mass':study.fe.mass(r['x']),
             'mass_port_squared':tail['mass_port_squared'],'controller_order':syn['controller'].nstates}
        row['full_pole_abscissa']=full_abscissa
        rows.append(row)
        quality.append({'design':name,'regularized_gap':abs(q['regularized_gap']),
                        'unrestricted_gap':abs(lower['gap']),'synthesis_slack':row['synthesis_slack'],
                        'kkt_residual':q['relative_kkt_residual'],
                        'coupling_complementarity':q['coupling_slack']['relative_complementarity'],
                        'brl_complementarity':q['brl_slack']['relative_complementarity'],
                        'coupling_sum_min':q['coupling_slack']['minimum_sum_eigenvalue'],
                        'brl_sum_min':q['brl_slack']['minimum_sum_eigenvalue'],
                        'constraint_rows':rank['rows'],'constraint_rank':rank['jacobian_rank'],
                        'constraint_min_singular':rank['minimum_singular_value'],
                        'physical_brl_max_eigenvalue':q['physical_brl_maximum_eigenvalue']})
        metadata[name]={'metrics':row,'regularized_synthesis':q,
                        'unrestricted_synthesis':{k:v for k,v in lower.items() if not k.startswith('_')},'regularity':rank}
        archive[name+'_x']=r['x']
        archive[name+'_phi']=mdl['phi']
        archive[name+'_Lambda']=mdl['Lambda']
        archive[name+'_Z']=syn['Z']
        archive[name+'_S']=syn['S']
        archive[name+'_unrestricted_Z']=lower['_Z']
        archive[name+'_unrestricted_S']=lower['_S']
        archive[name+'_unrestricted_gamma']=lower['primal']
        for key,v in zip(['X','Y','Ah','Bh','Ch','Dh'],lower['_x']): archive[name+'_unrestricted_'+key]=v
        for key,v in zip(['X','Y','Ah','Bh','Ch','Dh'],syn['x']): archive[name+'_'+key]=v
        for key in ['A','B','C','D']: archive[name+'_controller_'+key]=getattr(syn['controller'],key)
        archive[name+'_storage']=syn['storage']
        archive[name+'_gamma']=syn['gamma']
        print('final',row,flush=True)
        print('regularity',name,rank,flush=True)
    pd.DataFrame(rows).to_csv(DATA/'conic_results.csv',index=False)
    pd.DataFrame(quality).to_csv(DATA/'conic_quality.csv',index=False)
    np.savez(DATA/'conic_solutions.npz',**archive)
    cpu=platform.processor()
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if line.startswith('model name'):
            cpu=line.split(':',1)[1].strip()
            break
    meta={'settings':asdict(study.settings),'scales':asdict(study.scales),'config':asdict(study.cfg),
          'state_scale':study.state_scale,'platform':platform.platform(),'cpu':cpu,
          'patch':{'x':[.54,.60],'y':[.05,.15],'type':'fixed area average'},
          'results':metadata}
    (DATA/'conic_summary.json').write_text(json.dumps(meta,indent=2)+'\n')


def transfer_design(coarse,fine,x):
    raw=coarse.fe.full_design(x).reshape(coarse.cfg.nelx,coarse.cfg.nely)
    factor=fine.cfg.nelx//coarse.cfg.nelx
    fine_raw=np.repeat(np.repeat(raw,factor,axis=0),factor,axis=1).ravel()
    return fine.project(fine_raw[fine.fe.design])


def refinements(coarse,x):
    rows=[]
    for nx,ny in [(10,4),(20,8),(40,16)]:
        cfg=replace(base.CFG,nelx=nx,nely=ny)
        settings=replace(coarse.settings,max_iterations=8)
        fine=CertifiedStudy(cfg,coarse.scales,settings,coarse.state_scale)
        xf=x if nx==10 else transfer_design(coarse,fine,x)
        if nx==10:
            r=fine.evaluate(xf,gradient=True)
        else:
            r,h=fine.optimize(xf,True,f'mesh_{nx}')
            h.to_csv(DATA/f'mesh_{nx}_history.csv',index=False)
            np.savez(DATA/f'mesh_{nx}_design.npz',x=r['x'])
        mdl,syn,tail=r['modal'],r['synthesis'],r['tail']
        row={'nelx':nx,'nely':ny,'elements':fine.fe.ne,'dofs':len(fine.fe.free_dofs),
             'm':fine.m,'delta':tail['delta'],'eta_mu':syn['eta'],'delta_eta':tail['delta']*syn['eta'],
             'Gamma_mu':syn['eta']/r['margin'],'gap_normalized':mdl['gap'],
             'mass_port_squared':tail['mass_port_squared'],'volume':fine.fe.volume(r['x']),
             'mass':fine.fe.mass(r['x']),'eigen_seconds':mdl['eigen_seconds'],
             'sdp_seconds':syn['info']['sdp_seconds'],**r['timing']}
        rows.append(row)
        pd.DataFrame(rows).to_csv(DATA/'mesh_refinement.csv',index=False)
        print('mesh',row,flush=True)


def ablation(study,x):
    rows=[]
    for order in [8,12,16,20]:
        cfg=replace(study.cfg,retained_modes=order)
        s=CertifiedStudy(cfg,study.scales,study.settings,study.state_scale)
        r=s.evaluate(x)
        mdl,syn,tail=r['modal'],r['synthesis'],r['tail']
        nm=len(mdl['lam']); idx=np.r_[np.arange(order,nm),np.arange(nm+order,2*nm)]
        pf=s.plant(mdl,True)
        ts=ct.ss(pf[0][np.ix_(idx,idx)],pf[2][idx,:],np.vstack([pf[3][0,idx],pf[4][0,idx]]),np.zeros((2,1)))
        tn,_=hn(ts)
        rows.append({'m':order,'eta_mu':syn['eta'],'delta':tail['delta'],'tail_norm':tn,
                     'floor_delta':tail['floor_delta'],'margin':r['margin'],
                     'Gamma_mu':r['value'] if r['margin']>0 else np.nan})
        print('order',rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(DATA/'conic_ablation.csv',index=False)


def nyquist_baseline(study):
    """Circle-exclusion co-design adapted to the clamped velocity benchmark.

    Delissen et al. (2023), equations (3)--(5) and (13), give the local
    pole-circle construction. Here the fixed controller shape is a strictly
    proper velocity compensator, since this plant has no rigid-body mode.
    Full-model poles and the exact Hamiltonian norm are independent checks.
    """
    x0,_=base.initial_compliance_design(study.old)
    last={}; history=[]
    def evaluate(z):
        if 'z' in last and np.array_equal(z,last['z']): return last
        mdl=study.modal(z[:-2])
        gain,cutoff=np.exp(z[-2]),np.exp(z[-1])*study.scales.omega_scale
        lam=mdl['lam']; b=mdl['all_phi'].T@study.fe.force
        damp=study.scales.alpha+study.scales.beta*lam
        poles=-damp/2+1j*np.sqrt(lam-damp**2/4)
        residue=(study.cfg.force_reference/study.scales.vref)*b*b*poles/(poles-poles.conj())
        residue*=gain*cutoff/(poles+cutoff)
        def loop(s):
            den=lam[:,None]+damp[:,None]*s[None,:]+s[None,:]**2
            return gain*cutoff/(s+cutoff)*study.cfg.force_reference/study.scales.vref*np.sum(b[:,None]**2*s[None,:]/den,axis=0)
        ss=1j*poles.imag
        background=loop(ss)-residue/(-poles.real)
        centers=background-residue/(2*poles.real)
        radii=np.abs(residue)/(2*np.abs(poles.real))
        circles=np.abs(1+centers)-radii
        om=np.r_[0,np.geomspace(1e-2,3*np.sqrt(lam[-1]),1800),
                 (np.sqrt(lam)[:,None]*np.array([.97,.99,.997,1,1.003,1.01,1.03])).ravel()]
        mech=np.sum(b[:,None]**2/(lam[:,None]+damp[:,None]*(1j*om)+ (1j*om)**2),axis=0)
        q=study.cfg.force_reference/study.scales.qref*mech
        v=study.cfg.force_reference/study.scales.vref*1j*om*mech
        kval=-gain*cutoff/(1j*om+cutoff)
        cl=1/(1-kval*v)
        mat=np.zeros((len(om),2,2),complex)
        mat[:,0,0]=q*cl; mat[:,0,1]=q*kval*cl
        mat[:,1,0]=study.cfg.control_weight*kval*v*cl
        mat[:,1,1]=study.cfg.control_weight*kval*cl
        objective=float(np.linalg.svd(mat,compute_uv=False)[:,0].max())
        last.clear(); last.update(z=z.copy(),modal=mdl,objective=objective,
                                  circles=circles,gain=gain,cutoff=cutoff)
        return last
    def callback(z):
        r=evaluate(z)
        history.append({'iteration':len(history),'sampled_norm':r['objective'],
                        'minimum_circle_margin':float(r['circles'].min())})
        print('nyquist',history[-1],flush=True)
    z0=np.r_[x0,np.log(20.),np.log(1.)]
    result=minimize(lambda z:np.log(evaluate(z)['objective']),z0,method='SLSQP',
                    bounds=[(study.cfg.rho_min,1)]*len(x0)+[(-6,8),(np.log(.1),np.log(10))],
                    constraints=[{'type':'eq','fun':lambda z:study.fe.volume(z[:-2])-study.cfg.volume_fraction},
                                 {'type':'ineq','fun':lambda z:evaluate(z)['circles']-.5}],
                    callback=callback,options={'maxiter':24,'ftol':1e-7,'eps':3e-5,'disp':True})
    r=evaluate(result.x)
    ctrl=ct.ss([[-r['cutoff']/study.scales.omega_scale]],[[1]],
               [[-r['gain']*r['cutoff']/study.scales.omega_scale]],[[0]])
    p=study.plant(r['modal'])
    aug=closed_loop(p,ctrl)
    achieved,_=hn(aug)
    full=closed_loop(performance_plant(study.plant(r['modal'],True)),ctrl)
    full_norm,peak=hn(full)
    tail=study.tail(r['modal']); margin=1-tail['delta']*achieved
    zw,_=hn(aug[:2,:2]); zv,_=hn(aug[:2,3]); rw,_=hn(aug[2,:2]); uv,_=hn(aug[2,3])
    feedback_margin=1-tail['delta_v']*uv
    block=zw+(zv*tail['delta_v']+tail['delta_q'])*rw/feedback_margin
    om=np.r_[0,np.geomspace(1e-5,100*np.sqrt(r['modal']['lam'][-1]),12000)]
    b=r['modal']['all_phi'].T@study.fe.force
    lam=r['modal']['lam']; s=1j*om
    gv=study.cfg.force_reference/study.scales.vref*np.sum(b[:,None]**2*s[None,:]/
         (lam[:,None]+(study.scales.alpha+study.scales.beta*lam[:,None])*s[None,:]+s[None,:]**2),axis=0)
    loop=r['gain']*r['cutoff']/(s+r['cutoff'])*gv
    contour=np.r_[loop,loop[-2:0:-1].conj(),loop[:1]]
    winding=int(round(np.sum(np.angle((1+contour[1:])/(1+contour[:-1])))/(2*np.pi)))
    row={'design':'Nyquist','full_norm':full_norm,'achieved_eta':achieved,
         'delta':tail['delta'],'Gamma':achieved/margin,'small_gain_margin':margin,
         'minimum_circle_margin':float(r['circles'].min()),
         'sampled_modulus_margin':float(np.min(np.abs(1+loop))),
         'nyquist_winding':winding,'pole_abscissa':float(ct.poles(full).real.max()),
         'gain':r['gain'],'cutoff_rad_s':r['cutoff'],'controller_order':1,
         'mass':study.fe.mass(result.x[:-2]),'volume':study.fe.volume(result.x[:-2]),
         'optimizer_success':bool(result.success),'optimizer_message':str(result.message),
         'iterations':int(result.nit),'full_peak_rad_s':peak*study.scales.omega_scale}
    row.update(Gamma_block=block,feedback_margin=feedback_margin,
               delta_q=tail['delta_q'],delta_v=tail['delta_v'])
    assert winding==0 and row['pole_abscissa']<0 and row['minimum_circle_margin']>=.5-1e-5
    np.savez(DATA/'nyquist_design.npz',x=result.x[:-2],A=ctrl.A,B=ctrl.B,C=ctrl.C,D=ctrl.D)
    pd.DataFrame([row]).to_csv(DATA/'nyquist_results.csv',index=False)
    pd.DataFrame(history).to_csv(DATA/'nyquist_history.csv',index=False)
    print('nyquist final',row,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--max-iterations',type=int,default=SET.max_iterations)
    parser.add_argument('--report',action='store_true')
    parser.add_argument('--refine',action='store_true')
    parser.add_argument('--ablation',action='store_true')
    parser.add_argument('--nyquist',action='store_true')
    args=parser.parse_args()
    settings=replace(SET,max_iterations=args.max_iterations)
    study=CertifiedStudy(settings=settings)
    if args.nyquist:
        nyquist_baseline(study)
        return
    if args.report or args.refine or args.ablation:
        saved=np.load(DATA/'certified_designs.npz')
        if args.report:
            results={name:study.evaluate(saved[key],gradient=True) for name,key in [('Compliance','compliance'),('Augmented','augmented'),('Certified','certified')]}
            report(study,results)
        if args.refine: refinements(study,saved['certified'])
        if args.ablation: ablation(study,saved['certified'])
        return
    xcomp,_=base.initial_compliance_design(study.old)
    print('physical patch, scales',asdict(study.scales),'state scale',study.state_scale,flush=True)
    if args.probe:
        r=study.evaluate(xcomp,gradient=True)
        print('probe',r['value'],r['tail'],r['synthesis']['info'],flush=True)
        np.savez(DATA/'conic_probe.npz',x=xcomp,gradient=r['gradient'])
        return
    rcomp=study.evaluate(xcomp,gradient=True)
    aug,haug=study.optimize(xcomp,False,'augmented')
    cert,hcert=study.optimize(xcomp,True,'certified')
    haug.to_csv(DATA/'augmented_history.csv',index=False)
    hcert.to_csv(DATA/'certified_history.csv',index=False)
    np.savez(DATA/'certified_designs.npz',compliance=xcomp,augmented=aug['x'],certified=cert['x'])
    # Final reporting and refinement are separate functions for resumable runs.
    report(study,{'Compliance':rcomp,'Augmented':aug,'Certified':cert})


if __name__=='__main__':
    main()
