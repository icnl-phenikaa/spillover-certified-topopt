#!/usr/bin/env python3
"""Diagnostics: regularization sensitivity, single-threshold regularity,
sparse displacement-tail surrogate, and adaptive cluster windows.

This module writes numerical data only. It never writes manuscript text.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

import certified_studies as cs
import finite_element as base

DATA = cs.DATA
THRESHOLD = 1e-6  # one relative threshold for kernels, ranks and complementarity


def scaled(a):
    return cs.sym(a) / max(1., la.norm(a, 2))


def single_threshold_regularity(p, syn, settings, theta=THRESHOLD):
    """Kernel, rank and strict complementarity with one relative threshold.

    Slack and multiplier are each divided by max(1, spectral norm). The same
    theta defines the numerical kernel of the slack, the numerical rank of the
    multiplier, and the relative rank threshold of the compressed Jacobian.
    """
    n = len(p[0]); x = syn['x']
    Q = np.block([[x[0], np.eye(n)], [np.eye(n), x[1]]]) - settings.coupling_margin*np.eye(2*n)
    S = -cs.lmi_numpy(p, x, syn['gamma']) - settings.brl_margin*np.eye(2*n+7)
    blocks = []
    rows = []
    iu = np.triu_indices(n)

    def pack(parts):
        out = []
        for k, a in enumerate(parts):
            if k < 2:
                z = a[iu].copy(); z[iu[0] != iu[1]] *= np.sqrt(2); out.append(z)
            else:
                out.append(np.ravel(a))
        return np.concatenate(out)

    for slack, mult, which in [(Q, syn['S'], 'coupling'), (S, syn['Z'], 'brl')]:
        se, U = la.eigh(scaled(slack))
        me = la.eigvalsh(scaled(mult))
        kernel = se < theta
        active = int(np.sum(kernel)); mrank = int(np.sum(me > theta))
        below = se[kernel]; above = se[~kernel]
        mbelow = me[me <= theta]; mabove = me[me > theta]
        blocks.append({'block': which, 'dimension': len(se), 'slack_kernel': active,
                       'multiplier_rank': mrank,
                       'strictly_complementary': active == mrank,
                       'largest_kernel_slack': float(below.max()) if len(below) else None,
                       'smallest_positive_slack': float(above.min()) if len(above) else None,
                       'largest_null_multiplier': float(mbelow.max()) if len(mbelow) else None,
                       'smallest_positive_multiplier': float(mabove.min()) if len(mabove) else None,
                       'sum_minimum': float(la.eigvalsh(scaled(slack)+scaled(mult))[0])})
        V = U[:, kernel]
        for i in range(V.shape[1]):
            for j in range(i, V.shape[1]):
                seed = np.outer(V[:, i], V[:, j])
                if i != j: seed = (seed+seed.T)/np.sqrt(2)
                if which == 'coupling':
                    parts = cs.dual_coefficients(p, np.zeros_like(S), seed, 1.)
                else:
                    parts = cs.dual_coefficients(p, seed, np.zeros_like(Q), 1.)
                rows.append(pack(parts))
    J = np.asarray(rows)
    sv = la.svdvals(J)
    return {'theta': theta, 'blocks': blocks, 'rows': len(rows),
            'rank': int(np.sum(sv > theta*sv[0])),
            'sigma_min': float(sv[-1]), 'sigma_max': float(sv[0]),
            'sigma_ratio': float(sv[-1]/sv[0])}


def block_certificate(study, r):
    mdl, syn, tail = r['modal'], r['synthesis'], r['tail']
    p = study.plant(mdl)
    aug = cs.closed_loop(p, syn['controller'])
    achieved, _ = cs.hn(aug)
    zw, _ = cs.hn(aug[:2, :2]); zv, _ = cs.hn(aug[:2, 3]); rw, _ = cs.hn(aug[2, :2]); uv, _ = cs.hn(aug[2, 3])
    fm = 1 - tail['delta_v']*uv
    block = zw + (zv*tail['delta_v'] + tail['delta_q'])*rw/fm
    full = cs.closed_loop(cs.performance_plant(study.plant(mdl, True)), syn['controller'])
    full_norm, _ = cs.hn(full)
    return {'achieved_eta': achieved, 'Gamma_block': block, 'feedback_margin': fm,
            'Uv': uv, 'Zw': zw, 'Zv': zv, 'Rw': rw, 'full_norm': full_norm}


def diagnostics():
    saved = np.load(DATA/'certified_designs.npz')
    names = [('Compliance', 'compliance'), ('Augmented', 'augmented'), ('Certified', 'certified')]
    reference = cs.CertifiedStudy()
    out = {'threshold': THRESHOLD, 'designs': {}}
    for mu in [1e-5, 1e-6, 1e-7]:
        settings = replace(cs.SET, mu=mu)
        study = cs.CertifiedStudy(reference.cfg, reference.scales, settings, reference.state_scale)
        for name, key in names:
            r = study.evaluate(saved[key], gradient=True)
            p = study.plant(r['modal'])
            reg = single_threshold_regularity(p, r['synthesis'], settings)
            blk = block_certificate(study, r)
            row = {'mu': mu, 'eta_mu': r['synthesis']['eta'], 'gamma_mu': r['synthesis']['gamma'],
                   'Gamma_mu': r['value'], 'margin': r['margin'], 'delta': r['tail']['delta'],
                   'kkt': r['synthesis']['info']['relative_kkt_residual'],
                   'regularized_gap': r['synthesis']['info']['regularized_gap'],
                   'physical_brl': r['synthesis']['info']['physical_brl_maximum_eigenvalue'],
                   'regularity': reg, **blk}
            out['designs'].setdefault(name, {})[f'{mu:.0e}'] = row
            out['designs'][name].setdefault('_gradients', {})[f'{mu:.0e}'] = r['gradient']
            print(name, mu, {k: v for k, v in row.items() if k != 'regularity'}, flush=True)
            print('regularity', name, mu, reg, flush=True)
    for name, _ in names:
        grads = out['designs'][name].pop('_gradients')
        g0 = grads['1e-06']
        comp = {}
        for key, g in grads.items():
            comp[key] = {'relative_difference': float(la.norm(g-g0)/la.norm(g0)),
                         'cosine': float(g@g0/(la.norm(g)*la.norm(g0)))}
        out['designs'][name]['gradient_comparison'] = comp
        print(name, comp, flush=True)
    (DATA/'diagnostic_diagnostics.json').write_text(json.dumps(out, indent=2)+'\n')


def sparse_displacement_surrogate(study, mdl, theta, gradient_direction=None):
    """AM--GM surrogate h_q <= (1+1/(sqrt2 theta))/t + theta/(sqrt2 c^2).

    Returns the surrogate trace bound, evaluated with sparse solves only.
    """
    alpha, beta = study.scales.alpha, study.scales.beta
    K = sparse.csc_matrix(mdl['K']); M = sparse.csc_matrix(mdl['M'])
    C = sparse.csc_matrix(alpha*mdl['M']+beta*mdl['K'])
    b = study.fe.force
    phi = mdl['phi']; L = mdl['Lambda']
    br = phi.T@b
    kb = spla.spsolve(K, b); cb = spla.spsolve(C, b)
    flex = b@kb - br@la.solve(L, br)
    R = la.inv(alpha*np.eye(study.m)+beta*L)
    second = cb@(M@cb) - br@R@R@br
    a1 = 1 + 1/(np.sqrt(2)*theta); a2 = theta/np.sqrt(2)
    aq = study.cfg.force_reference/study.scales.qref
    return {'delta_q_sparse': float(aq*(a1*flex+a2*second)), 'flex_trace': float(flex),
            'damping_trace': float(second), 'a1': a1, 'a2': a2}


def sparse_tail_study():
    rows = []
    coarse = cs.CertifiedStudy()
    designs = {10: np.load(DATA/'certified_designs.npz')['certified'],
               20: np.load(DATA/'mesh_20_design.npz')['x'],
               40: np.load(DATA/'mesh_40_design.npz')['x']}
    zeta_ref = coarse.cfg.damping_ratio
    for (nx, ny) in [(10, 4), (20, 8), (40, 16)]:
        cfg = replace(base.CFG, nelx=nx, nely=ny)
        st = cs.CertifiedStudy(cfg, coarse.scales, cs.SET, coarse.state_scale)
        x = designs[nx]
        mdl = st.modal(x)
        tail = st.tail(mdl)
        # Sparse retained cluster: shift-invert Lanczos for m+1 eigenpairs.
        K = sparse.csc_matrix(mdl['K']); M = sparse.csc_matrix(mdl['M'])
        t0 = time.perf_counter()
        vals, vecs = spla.eigsh(K, k=st.m+1, M=M, sigma=0, which='LM')
        t_eig = time.perf_counter()-t0
        order = np.argsort(vals); vals = vals[order]
        sparse_mdl = dict(mdl); sparse_mdl['phi'] = vecs[:, order[:st.m]]
        sparse_mdl['Lambda'] = np.diag(vals[:st.m])
        t0 = time.perf_counter()
        sur = sparse_displacement_surrogate(st, sparse_mdl, 2*zeta_ref)
        t_sur = time.perf_counter()-t0
        lam = mdl['lam']; dense_err = float(np.max(np.abs(vals[:st.m+1]-lam[:st.m+1])/lam[:st.m+1]))
        ratio = sur['delta_q_sparse']/tail['delta_q']
        delta_sparse = float(np.hypot(sur['delta_q_sparse'], tail['delta_v']))
        thetas = np.geomspace(1e-3, 1, 61)
        best = min(thetas, key=lambda th: sparse_displacement_surrogate(st, sparse_mdl, th)['delta_q_sparse'])
        best_val = sparse_displacement_surrogate(st, sparse_mdl, best)['delta_q_sparse']
        # Relative gaps in a window of cluster boundaries.
        window = range(16, 29)
        rel = {k: float((lam[k]-lam[k-1])/lam[k-1]) for k in window}
        kstar = max(rel, key=rel.get)
        row = {'nelx': nx, 'dofs': len(st.fe.free_dofs), 'delta_q_exact': tail['delta_q'],
               'delta_q_sparse': sur['delta_q_sparse'], 'sparse_to_exact': float(ratio),
               'delta_exact': tail['delta'], 'delta_sparse': delta_sparse,
               'theta': 2*zeta_ref, 'best_theta': float(best), 'best_ratio': float(best_val/tail['delta_q']),
               'dense_eigensolve_seconds': mdl['eigen_seconds'], 'sparse_eigensolve_seconds': t_eig,
               'sparse_surrogate_seconds': t_sur, 'eigenvalue_relative_error': dense_err,
               'gap_normalized_m20': mdl['gap'], 'relative_gap_m20': rel[20],
               'window_best_m': int(kstar), 'window_best_relative_gap': rel[kstar],
               'relative_gaps': rel}
        rows.append(row)
        print('sparse', row, flush=True)
    (DATA/'diagnostic_sparse_tail.json').write_text(json.dumps(rows, indent=2)+'\n')


def sparse_derivative_check():
    """Finite-difference check of the sparse surrogate trace derivative."""
    study = cs.CertifiedStudy()
    x = np.load(DATA/'rp_certified_design.npz')['x']
    rng = np.random.default_rng(20261002)
    d = rng.normal(size=len(x)); d[(x < .051) | (x > .999)] = 0; d /= la.norm(d)
    alpha, beta = study.scales.alpha, study.scales.beta
    theta = 2*study.cfg.damping_ratio

    def value(z):
        mdl = study.modal(z)
        return sparse_displacement_surrogate(study, mdl, theta)['delta_q_sparse']
    mdl = study.modal(x)
    K, M = mdl['K'], mdl['M']; C = alpha*M+beta*K
    b = study.fe.force; phi = mdl['phi']; L = mdl['Lambda']
    kd, md = study.fe.directional_matrices(x, d)
    cd = alpha*md+beta*kd
    kb = la.solve(K, b); cb = la.solve(C, b)
    lamb, U = la.eigh(L); phid = phi@U; br = phid.T@b
    # Exact derivative of the retained parts through eigenpair derivatives.
    dl = np.array([phid[:, i]@(kd-lamb[i]*md)@phid[:, i] for i in range(study.m)])
    a1 = 1+1/(np.sqrt(2)*theta); a2 = theta/np.sqrt(2)
    aq = study.cfg.force_reference/study.scales.qref
    # Full spectral evaluation of d/ds [b^T Phi f(Lambda) Phi^T b] by divided differences.
    lam, ap = la.eigh(K, M)
    ab = ap.T@b
    def retained_derivative(f, fp):
        w = np.zeros_like(lam); wp = np.zeros_like(lam)
        w[:study.m] = f(lam[:study.m]); wp[:study.m] = fp(lam[:study.m])
        F = cs.divided_difference(lam, w, wp)
        G = cs.divided_difference(lam, lam*w, w+lam*wp)
        E_K = ap.T@kd@ap; E_M = ap.T@md@ap
        return float(ab@((F*E_K-G*E_M)@ab))
    dflex = -kb@kd@kb - retained_derivative(lambda t: 1/t, lambda t: -1/t**2)
    ci = la.solve(C, M@cb)
    dsecond = cb@md@cb - 2*cb@cd@ci
    c2 = lambda t: (alpha+beta*t)**-2
    c2p = lambda t: -2*beta*(alpha+beta*t)**-3
    dsecond -= retained_derivative(c2, c2p)
    analytic = aq*(a1*dflex+a2*dsecond)
    out = []
    for h in [1e-3, 1e-4, 1e-5]:
        fd = (value(x+h*d)-value(x-h*d))/(2*h)
        out.append({'step': h, 'fd': fd, 'analytic': analytic, 'relative_error': abs(fd-analytic)/abs(analytic)})
        print('sparse derivative', out[-1], flush=True)
    (DATA/'diagnostic_sparse_derivative.json').write_text(json.dumps(out, indent=2)+'\n')


def cluster_alternatives():
    """Re-evaluate the finest final design at cluster boundaries near m=20."""
    coarse = cs.CertifiedStudy()
    x = np.load(DATA/'mesh_40_design.npz')['x']
    rows = []
    for m in [19, 20, 21]:
        cfg = replace(base.CFG, nelx=40, nely=16, retained_modes=m)
        st = cs.CertifiedStudy(cfg, coarse.scales, cs.SET, coarse.state_scale)
        r = st.evaluate(x)
        lam = r['modal']['lam']
        row = {'m': m, 'relative_gap': float((lam[m]-lam[m-1])/lam[m-1]),
               'gap_normalized': r['modal']['gap'], 'eta_mu': r['synthesis']['eta'],
               'gamma_mu': r['synthesis']['gamma'], 'delta': r['tail']['delta'],
               'delta_eta': r['tail']['delta']*r['synthesis']['eta'], 'Gamma_mu': r['value'],
               'sdp_seconds': r['synthesis']['info']['sdp_seconds']}
        rows.append(row)
        print('cluster', row, flush=True)
    (DATA/'diagnostic_clusters.json').write_text(json.dumps(rows, indent=2)+'\n')


def threshold_sweep():
    """Nondegeneracy rank test as the single threshold varies (mu = 1e-6)."""
    saved = np.load(DATA/'certified_designs.npz')
    study = cs.CertifiedStudy()
    out = {}
    for name, key in [('Compliance', 'compliance'), ('Augmented', 'augmented'), ('Certified', 'certified')]:
        r = study.evaluate(saved[key])
        p = study.plant(r['modal'])
        out[name] = []
        for theta in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
            reg = single_threshold_regularity(p, r['synthesis'], study.settings, theta)
            row = {'theta': theta, 'rows': reg['rows'], 'rank': reg['rank'], 'sigma_ratio': reg['sigma_ratio'],
                   'kernels': [b['slack_kernel'] for b in reg['blocks']],
                   'multiplier_ranks': [b['multiplier_rank'] for b in reg['blocks']]}
            out[name].append(row)
            print(name, row, flush=True)
    (DATA/'diagnostic_threshold_sweep.json').write_text(json.dumps(out, indent=2)+'\n')


class RejectingStudy(cs.CertifiedStudy):
    """A failed conic solve at a trial design is treated as a rejected candidate."""
    failures = 0

    def evaluate(self, x, gradient=False, certified=True, anchor=None):
        try:
            return super().evaluate(x, gradient, certified, anchor)
        except Exception as error:
            if anchor is None: raise
            RejectingStudy.failures += 1
            print('rejected candidate after solver failure:', error, flush=True)
            mdl = self.modal(x, anchor)
            return {'x': np.asarray(x).copy(), 'modal': mdl, 'margin': -1., 'value': np.inf,
                    'synthesis': {'info': {'pole_abscissa': 1., 'physical_brl_maximum_eigenvalue': 1.}}}


def mu_search(mu):
    """Repeat the certified density search at another regularization parameter."""
    settings = replace(cs.SET, mu=mu)
    study = RejectingStudy(settings=settings)
    x0 = np.load(DATA/'certified_designs.npz')['compliance']
    r, h = study.optimize(x0, True, f'certified_mu{mu:.0e}')
    h.to_csv(DATA/f'certified_mu{mu:.0e}_history.csv', index=False)
    blk = block_certificate(study, r)
    ref = np.load(DATA/'certified_designs.npz')['certified']
    row = {'mu': mu, 'Gamma_mu': r['value'], 'eta_mu': r['synthesis']['eta'], 'gamma_mu': r['synthesis']['gamma'],
           'delta': r['tail']['delta'], 'iterations': len(h)-1, 'projected_residual': float(h['projected_residual'].iloc[-1]),
           'design_difference_max': float(np.max(np.abs(r['x']-ref))),
           'design_difference_rel': float(la.norm(r['x']-ref)/la.norm(ref)),
           'rejected_solver_failures': RejectingStudy.failures, **blk}
    print('mu search', row, flush=True)
    (DATA/f'certified_mu{mu:.0e}_result.json').write_text(json.dumps(row, indent=2)+'\n')
    np.savez(DATA/f'certified_mu{mu:.0e}_design.npz', x=r['x'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold-sweep', action='store_true')
    parser.add_argument('--mu-search', type=float, default=None)
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--sparse-derivative', action='store_true')
    parser.add_argument('--clusters', action='store_true')
    a = parser.parse_args()
    if a.threshold_sweep: threshold_sweep()
    if a.mu_search is not None: mu_search(a.mu_search)
    if a.diagnostics: diagnostics()
    if a.sparse: sparse_tail_study()
    if a.sparse_derivative: sparse_derivative_check()
    if a.clusters: cluster_alternatives()


if __name__ == '__main__':
    main()
