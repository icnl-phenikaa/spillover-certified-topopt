#!/usr/bin/env python3
"""Scaled robust-performance certificate as a single conic objective.

For fixed d>0 and residual bound delta, the condition
    || diag(G^{-1/2} I, d sqrt(delta) I) N diag(G^{-1/2} I, sqrt(delta)/d I) ||_inf < 1
implies ||F_u(N, Delta)||_inf < G for every ||Delta||_inf <= delta.
After congruence it is the full-order bounded-real LMI with the two -gamma I
blocks replaced by -diag(G, G, d^2/delta, d^2/delta) and -diag(G, G, 1/(d^2 delta)).
It is affine in (chi, G), so minimizing G is one conic solve.

This module writes numerical data only. It never writes manuscript text.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.linalg as la

import certified_studies as cs
import finite_element as base

DATA = cs.DATA
PERF_IN = np.array([1., 1., 0., 0.])   # [d, n | p_q, p_v]
PERF_OUT = np.array([1., 1., 0.])      # [z_q, w_u u | r]


def lmi_rp(p, x, gamma, cin, cout):
    A, B1, B2, C1, C2, D11, D12, D21 = p
    L = cs.lmi_numpy(p, x, 0.)
    n = len(A); nd = B1.shape[1]
    k = slice(2*n, 2*n+nd); l = slice(2*n+nd, None)
    L[k, k] -= np.diag(gamma*PERF_IN+cin*(1-PERF_IN))
    L[l, l] -= np.diag(gamma*PERF_OUT+cout*(1-PERF_OUT))
    return L


def dual_coefficients_rp(p, Z, S, gamma_multiplier):
    """Lagrangian gradient; only the gamma entry differs from the scalar problem."""
    coeff = cs.dual_coefficients(p, Z, S, gamma_multiplier)
    n, nd = len(p[0]), p[1].shape[1]
    k = slice(2*n, 2*n+nd); l = slice(2*n+nd, None)
    coeff[-1] = np.array(1-np.sum(np.diag(Z[k, k])*PERF_IN)-np.sum(np.diag(Z[l, l])*PERF_OUT)-gamma_multiplier)
    return coeff


class RPSynthesis(cs.ConicSynthesis):
    def __init__(self, n, settings=cs.SET):
        self.n, self.settings = n, settings
        self.params = [cp.Parameter(shape) for shape in [(n, n), (n, 4), (n, 1), (3, n), (1, n), (3, 4), (3, 1), (1, 4)]]
        self.cin = cp.Parameter(nonneg=True); self.cout = cp.Parameter(nonneg=True)
        A, B1, B2, C1, C2, D11, D12, D21 = self.params
        self.x = [cp.Variable((n, n), symmetric=True), cp.Variable((n, n), symmetric=True),
                  cp.Variable((n, n)), cp.Variable((n, 1)), cp.Variable((1, n)), cp.Variable((1, 1))]
        X, Y, Ah, Bh, Ch, Dh = self.x
        self.gamma = cp.Variable()
        aa = A@X+B2@Ch; bb = Y@A+Bh@C2; ab = Ah.T+A+B2@Dh@C2; ac = B1+B2@Dh@D21
        ad = (C1@X+D12@Ch).T; bc = Y@B1+Bh@D21; bd = (C1+D12@Dh@C2).T; cd = (D11+D12@Dh@D21).T
        din = self.gamma*np.diag(PERF_IN)+self.cin*np.diag(1-PERF_IN)
        dout = self.gamma*np.diag(PERF_OUT)+self.cout*np.diag(1-PERF_OUT)
        self.L = cp.bmat([[aa+aa.T, ab, ac, ad], [ab.T, bb+bb.T, bc, bd], [ac.T, bc.T, -din, cd], [ad.T, bd.T, cd.T, -dout]])
        self.Q = cp.bmat([[X, np.eye(n)], [np.eye(n), Y]])
        self.reg = sum(cp.sum_squares(v) for v in self.x+[self.gamma])/2
        self.constraints = [self.Q >> settings.coupling_margin*np.eye(2*n),
                            self.L << -settings.brl_margin*np.eye(2*n+7), self.gamma >= 0]
        self.problem = cp.Problem(cp.Minimize(self.gamma+settings.mu*self.reg), self.constraints)

    def solve(self, plant, cin, cout):
        start = time.perf_counter()
        for p, v in zip(self.params, plant): p.value = v
        self.cin.value, self.cout.value = cin, cout
        self.problem.solve(solver=cp.MOSEK, warm_start=True, ignore_dpp=True, mosek_params=cs.SOLVER_OPTIONS)
        if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError('rp synthesis status '+self.problem.status)
        x = [v.value.copy() for v in self.x]
        gamma = float(self.gamma.value); value = float(self.problem.value)
        Z = cs.positive(self.constraints[1].dual_value); S = cs.positive(self.constraints[0].dual_value)
        z0 = max(0., float(self.constraints[2].dual_value))
        coeff = dual_coefficients_rp(plant, Z, S, z0)
        n = self.n
        zero = [np.zeros_like(v) for v in x]
        L0 = lmi_rp(plant, zero, 0., cin, cout)+self.settings.brl_margin*np.eye(2*n+7)
        Q0 = np.block([[-self.settings.coupling_margin*np.eye(n), np.eye(n)],
                       [np.eye(n), -self.settings.coupling_margin*np.eye(n)]])
        dual = float(np.sum(Z*L0)-np.sum(S*Q0)-sum(np.sum(c*c) for c in coeff)/(2*self.settings.mu))
        stationarity = [c+self.settings.mu*v for c, v in zip(coeff, x+[np.array(gamma)])]
        kkt = float(np.sqrt(sum(np.sum(v*v) for v in stationarity))/max(1, np.sqrt(sum(np.sum(c*c) for c in coeff))))
        ctrl, P = cs.recover(plant, x)
        cl = cs.closed_loop(plant, ctrl)
        a, b, c, d = cl.A, cl.B, cl.C, cl.D
        din = np.diag(gamma*PERF_IN+cin*(1-PERF_IN)); dout = np.diag(gamma*PERF_OUT+cout*(1-PERF_OUT))
        brl = np.block([[a.T@P+P@a, P@b, c.T], [b.T@P, -din, d.T], [c, d, -dout]])
        n = self.n; nd = 4
        k = slice(2*n, 2*n+nd); l = slice(2*n+nd, None)
        info = {'objective': value, 'gamma': gamma, 'regularized_dual': dual, 'regularized_gap': value-dual,
                'relative_kkt_residual': kkt,
                'physical_brl_maximum_eigenvalue': float(la.eigvalsh(cs.sym(brl))[-1]),
                'minimum_storage_eigenvalue': float(la.eigvalsh(P)[0]),
                'pole_abscissa': float(np.max(la.eigvals(cl.A).real)), 'sdp_seconds': time.perf_counter()-start,
                'residual_input_trace': float(np.trace(Z[k, k][2:, 2:])), 'residual_output_trace': float(Z[l, l][2, 2])}
        return {'x': x, 'gamma': gamma, 'eta': value, 'Z': Z, 'S': S, 'controller': ctrl, 'storage': P, 'info': info}

    def lower_bound(self, plant, cin, cout):
        for p, v in zip(self.params, plant): p.value = v
        self.cin.value, self.cout.value = cin, cout
        prob = cp.Problem(cp.Minimize(self.gamma), [self.Q >> 0, self.L << 0, self.gamma >= 0])
        prob.solve(solver=cp.MOSEK, ignore_dpp=True, mosek_params=cs.SOLVER_OPTIONS)
        return float(prob.value)


def contract(study, mdl, Kseed, Mseed):
    ke = np.zeros(study.fe.ne); me = np.zeros(study.fe.ne)
    gk = np.zeros((study.fe.ndof, study.fe.ndof)); gm = np.zeros_like(gk)
    free = np.ix_(study.fe.free_dofs, study.fe.free_dofs)
    gk[free], gm[free] = Kseed, Mseed
    for e, ed in enumerate(study.fe.edofs):
        ke[e] = np.sum(gk[np.ix_(ed, ed)]*study.fe.ke_unit)*study.cfg.young*(1-study.cfg.e_floor)*study.cfg.simp_p*mdl['rho'][e]**(study.cfg.simp_p-1)
        me[e] = np.sum(gm[np.ix_(ed, ed)]*study.fe.me_unit)*study.cfg.material_density
    return study.physical_jacobian().T@(ke+me)


class RPStudy(cs.CertifiedStudy):
    """Density objective = regularized robust-performance level at fixed scaling d."""
    scaling = 2.

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.rp = RPSynthesis(2*self.m, self.settings)

    def coefficients(self, delta):
        d2 = self.scaling**2
        return d2/delta, 1/(d2*delta)

    def evaluate(self, x, gradient=False, certified=True, anchor=None):
        mdl = self.modal(x, anchor)
        tail = self.tail(mdl)
        cin, cout = self.coefficients(tail['delta'])
        try:
            syn = self.rp.solve(self.plant(mdl), cin, cout)
        except Exception:
            if anchor is None: raise
            # A failed conic solve at a trial design rejects the trial.
            return {'x': np.asarray(x).copy(), 'modal': mdl, 'tail': tail, 'margin': -1., 'value': np.inf,
                    'synthesis': {'info': {'pole_abscissa': 1., 'physical_brl_maximum_eigenvalue': 1.}}}
        # The RP inequality itself certifies the full loop; keep a nominal margin field.
        result = {'x': np.asarray(x).copy(), 'modal': mdl, 'synthesis': syn, 'tail': tail,
                  'margin': 1., 'value': float(syn['eta'])}
        if gradient:
            result['gradient'], result['timing'] = self.structural_gradient(mdl, syn, True)
        return result

    def delta_coefficient(self, syn, delta):
        d2 = self.scaling**2
        return d2/delta**2*syn['info']['residual_input_trace']+syn['info']['residual_output_trace']/(d2*delta**2)

    def spectral_gradient(self, mdl, syn):
        """Reference implementation: residual derivative by full divided differences."""
        retained, _ = cs.CertifiedStudy.structural_gradient(self, mdl, syn, False)
        tail = self.tail(mdl, gradient=True)
        return retained+self.delta_coefficient(syn, tail['delta'])*contract(self, mdl, tail['Kseed'], tail['Mseed'])

    def structural_gradient(self, mdl, syn, certified=True):
        """Retained conic adjoint plus c_delta * D delta, velocity part folded sparsely."""
        import scipy.sparse as sparse
        import scipy.sparse.linalg as spla
        p = self.plant(mdl)
        a, b1, b2, c1, c2 = cs.lmi_seeds(p, syn['x'], syn['Z'])
        m = self.m; w = self.scales.omega_scale; st = self.state_scale
        barL = cs.sym(-a[m:, :m]/w**2-self.scales.beta*a[m:, m:]/w)
        barb = (b1[m:, 0]+b2[m:, 0])*st*self.cfg.force_reference/(w*self.scales.vref)+(c1[0, :m]+c2[0, m:])/st
        phi = mdl['phi']; lr = mdl['Lambda']; force = self.fe.force[:, None]
        br = phi.T@force
        W = force@barb[None, :]
        tail_start = time.perf_counter()
        tail = self.tail(mdl)
        cdelta = self.delta_coefficient(syn, tail['delta'])
        vcoef = cdelta*tail['delta_v']*self.cfg.force_reference/(tail['delta']*self.scales.vref)
        R = la.inv(self.scales.alpha*np.eye(m)+self.scales.beta*lr)
        barL += self.scales.beta*vcoef*R@br@br.T@R
        W -= 2*vcoef*force@br.T@R
        invseed = vcoef*np.outer(mdl['_c_solution'], mdl['_c_solution'])
        directK = -self.scales.beta*invseed; directM = -self.scales.alpha*invseed
        tail_seconds = time.perf_counter()-tail_start
        theta = cs.sym(lr@barL)+.5*cs.sym(phi.T@W)
        deflated_start = time.perf_counter()
        lamb, U = la.eigh(lr)
        phid = phi@U; Wd = W@U
        Xi = np.zeros_like(phi)
        mphi = mdl['M']@phid
        for i in range(m):
            shift = sparse.csc_matrix(mdl['K']-lamb[i]*mdl['M'])
            saddle = sparse.bmat([[shift, sparse.csc_matrix(mphi)],
                                  [sparse.csc_matrix(mphi.T), sparse.csc_matrix((m, m))]], format='csc')
            rhs = np.r_[Wd[:, i]-mphi@(phid.T@Wd[:, i]), np.zeros(m)]
            Xi[:, i] = spla.spsolve(saddle, rhs)[:len(phi)]
        Xi = Xi@U.T
        deflated_seconds = time.perf_counter()-deflated_start
        Kseed = cs.sym(phi@barL@phi.T-Xi@phi.T)+directK
        Mseed = cs.sym(-phi@theta@phi.T+Xi@lr@phi.T)+directM
        tail_start = time.perf_counter()
        q, qp, _, _ = cs.rayleigh_weights(mdl['lam'], self.scales.alpha, self.scales.beta)
        q[:m] = 0; qp[:m] = 0
        ap = mdl['all_phi']; ab = ap.T@self.fe.force
        qcoef = cdelta*tail['delta_q']*self.cfg.force_reference/(tail['delta']*self.scales.qref)
        rankseed = np.outer(ab, ab)
        F = cs.divided_difference(mdl['lam'], q, qp)
        G = cs.divided_difference(mdl['lam'], mdl['lam']*q, q+mdl['lam']*qp)
        Kseed += qcoef*cs.sym(ap@(F*rankseed)@ap.T)
        Mseed -= qcoef*cs.sym(ap@(G*rankseed)@ap.T)
        tail_seconds += time.perf_counter()-tail_start
        start = time.perf_counter()
        grad = contract(self, mdl, Kseed, Mseed)
        return grad, {'deflated_seconds': deflated_seconds, 'tail_gradient_seconds': tail_seconds,
                      'damping_solve_seconds': mdl['_c_seconds'], 'contraction_seconds': time.perf_counter()-start}


def rp_level(study, r, ctrl=None, d=None):
    """Smallest G with ||D_l N D_r|| <= 1 for a fixed controller (bisection)."""
    d = study.scaling if d is None else d
    p = study.plant(r['modal']); ctrl = ctrl or r['synthesis']['controller']
    aug = cs.closed_loop(p, ctrl); delta = r['tail']['delta']
    def scaled_norm(G):
        dl = np.r_[np.full(2, G**-.5), d*np.sqrt(delta)]
        dr = np.r_[np.full(2, G**-.5), np.full(2, np.sqrt(delta)/d)]
        s = cs.ct.ss(aug.A, aug.B*dr[None, :], dl[:, None]*aug.C, dl[:, None]*aug.D*dr[None, :])
        return cs.hn(s)[0]
    lo, hi = 1., 100.
    for _ in range(50):
        mid = np.sqrt(lo*hi)
        if scaled_norm(mid) < 1: hi = mid
        else: lo = mid
    return hi


def summary(study, name, r):
    mdl, syn, tail = r['modal'], r['synthesis'], r['tail']
    p = study.plant(mdl)
    aug = cs.closed_loop(p, syn['controller'])
    full = cs.closed_loop(cs.performance_plant(study.plant(mdl, True)), syn['controller'])
    ab = float(cs.ct.poles(full).real.max()); fn = cs.hn(full)[0]
    zw, _ = cs.hn(aug[:2, :2]); zv, _ = cs.hn(aug[:2, 3]); rw, _ = cs.hn(aug[2, :2]); uv, _ = cs.hn(aug[2, 3])
    fm = 1-tail['delta_v']*uv
    blk = zw+(zv*tail['delta_v']+tail['delta_q'])*rw/fm
    cin, cout = study.coefficients(tail['delta'])
    row = {'design': name, 'Gamma_rp_mu': syn['eta'], 'Gamma_rp': syn['gamma'],
           'Gamma_rp_lb': study.rp.lower_bound(p, cin, cout), 'Gamma_rp_achieved': rp_level(study, r),
           'Gamma_block': blk, 'full_norm': fn, 'retained_norm': zw, 'delta': tail['delta'],
           'Uv': uv, 'feedback_margin': fm, 'full_pole_abscissa': ab, 'gap_normalized': mdl['gap'],
           'physical_brl': syn['info']['physical_brl_maximum_eigenvalue'],
           'volume': study.fe.volume(r['x']), 'mass': study.fe.mass(r['x'])}
    print('rp', row, flush=True)
    return row


def gradient_check(study, x):
    r = study.evaluate(x, gradient=True)
    rng = np.random.default_rng(20261004)
    out = []
    for _ in range(3):
        d = rng.normal(size=len(x)); d[(x < .051) | (x > .999)] = 0
        v = study.volume_gradient.copy(); v[(x < .051) | (x > .999)] = 0
        d -= v*(v@d)/(v@v); d /= la.norm(d)
        h = 3e-4
        fd = (study.evaluate(x+h*d)['value']-study.evaluate(x-h*d)['value'])/(2*h)
        ad = r['gradient']@d
        out.append({'fd': fd, 'adjoint': ad, 'relative_error': abs(fd-ad)/max(1, abs(fd), abs(ad))})
        print('rp gradient', out[-1], flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--scaling', type=float, default=2.)
    a = parser.parse_args()
    RPStudy.scaling = a.scaling
    study = RPStudy()
    saved = np.load(DATA/'certified_designs.npz')
    if a.evaluate:
        rows = [summary(study, name, study.evaluate(saved[key])) for name, key in
                [('Compliance', 'compliance'), ('Augmented', 'augmented'), ('Certified', 'certified')]]
        nyq = np.load(DATA/'nyquist_design.npz')
        rows.append(summary(study, 'Nyquist density', study.evaluate(nyq['x'])))
        pd.DataFrame(rows).to_csv(DATA/f'rp_evaluation_d{a.scaling:g}.csv', index=False)
    if a.check:
        res = gradient_check(study, saved['compliance'])
        (DATA/'rp_gradient_check.json').write_text(json.dumps(res, indent=2)+'\n')
    if a.run:
        r, h = study.optimize(saved['compliance'], True, 'rp_certified')
        h.to_csv(DATA/'rp_certified_history.csv', index=False)
        np.savez(DATA/'rp_certified_design.npz', x=r['x'])
        row = summary(study, 'RP certified', r)
        pd.DataFrame([row]).to_csv(DATA/'rp_certified_results.csv', index=False)


if __name__ == '__main__':
    main()
