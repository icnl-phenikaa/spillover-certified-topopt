#!/usr/bin/env python3
"""Robust-performance studies reported in the manuscript.

Run after `rp_study.py --run`. This module writes numerical data only and
never writes manuscript text.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import scipy.linalg as la

import certified_studies as cs
import finite_element as base
import noncollocated_study as nc
import diagnostic_studies as rv
import rp_study as rp

DATA = cs.DATA


def designs():
    saved = np.load(DATA/'certified_designs.npz')
    return {'Compliance': saved['compliance'], 'Augmented': saved['augmented'],
            'RP certified': np.load(DATA/'rp_certified_design.npz')['x'],
            'Nyquist': np.load(DATA/'nyquist_design.npz')['x']}


def blocks(aug):
    zw, _ = cs.hn(aug[:2, :2]); zv, _ = cs.hn(aug[:2, 3]); rw, _ = cs.hn(aug[2, :2]); uv, _ = cs.hn(aug[2, 3])
    n12, _ = cs.hn(aug[:2, 2:]); n21, _ = cs.hn(aug[2:, :2])
    return zw, zv, rw, uv, n12, n21


def report():
    study = rp.RPStudy()
    scalar = cs.CertifiedStudy()
    rows, archive = [], {}
    nyq = np.load(DATA/'nyquist_design.npz')
    nyq_ctrl = cs.ct.ss(nyq['A'], nyq['B'], nyq['C'], nyq['D'])
    for name, x in designs().items():
        r = study.evaluate(x, gradient=True)
        mdl, syn, tail = r['modal'], r['synthesis'], r['tail']
        p = study.plant(mdl)
        aug = cs.closed_loop(p, syn['controller'])
        full = cs.closed_loop(cs.performance_plant(study.plant(mdl, True)), syn['controller'])
        zw, zv, rw, uv, n12, n21 = blocks(aug)
        cin, cout = study.coefficients(tail['delta'])
        row = {'design': name, 'Gamma_rp_mu': syn['eta'], 'Gamma_rp': syn['gamma'],
               'Gamma_rp_lb': study.rp.lower_bound(p, cin, cout), 'Gamma_rp_achieved': rp.rp_level(study, r),
               'full_norm': cs.hn(full)[0], 'full_pole_abscissa': float(cs.ct.poles(full).real.max()),
               'Gamma_block': zw+(zv*tail['delta_v']+tail['delta_q'])*rw/(1-tail['delta_v']*uv),
               'feedback_margin': 1-tail['delta_v']*uv, 'Uv': uv, 'N12': n12, 'N21': n21,
               'perron_scaling': float(np.sqrt(n12/n21)), 'achieved_eta': cs.hn(aug)[0],
               'delta': tail['delta'], 'delta_q': tail['delta_q'], 'delta_v': tail['delta_v'],
               'tail_norm': nc.tail_norm(study, mdl), 'floor_delta': tail['floor_delta'],
               'gap_normalized': mdl['gap'], 'mass': study.fe.mass(x), 'volume': study.fe.volume(x),
               'regularized_gap': syn['info']['regularized_gap'], 'kkt': syn['info']['relative_kkt_residual'],
               'physical_brl': syn['info']['physical_brl_maximum_eigenvalue'],
               'storage_min': syn['info']['minimum_storage_eigenvalue'],
               'sdp_seconds': syn['info']['sdp_seconds'], 'eigen_seconds': mdl['eigen_seconds'], **r['timing']}
        # Scalar synthesis on the same topology: eta_mu is the sharp scalar certificate.
        s = scalar.evaluate(x)
        saug = cs.closed_loop(scalar.plant(s['modal']), s['synthesis']['controller'])
        sfull = cs.closed_loop(cs.performance_plant(scalar.plant(s['modal'], True)), s['synthesis']['controller'])
        row.update(eta_mu=s['synthesis']['eta'], gamma_mu=s['synthesis']['gamma'],
                   scalar_achieved=cs.hn(saug)[0], scalar_full_norm=cs.hn(sfull)[0],
                   scalar_lb=scalar.synthesis.lower_bound(scalar.plant(s['modal']))['dual'],
                   fractional=s['synthesis']['eta']/s['margin'])
        if name == 'Nyquist':
            naug = cs.closed_loop(p, nyq_ctrl)
            nzw, nzv, nrw, nuv, _, _ = blocks(naug)
            nfull = cs.closed_loop(cs.performance_plant(study.plant(mdl, True)), nyq_ctrl)
            row.update(nyquist_controller_eta=cs.hn(naug)[0], nyquist_controller_full=cs.hn(nfull)[0],
                       nyquist_controller_rp=rp.rp_level(study, r, ctrl=nyq_ctrl),
                       nyquist_controller_block=nzw+(nzv*tail['delta_v']+tail['delta_q'])*nrw/(1-tail['delta_v']*nuv))
        rows.append(row)
        for key in ['A', 'B', 'C', 'D']: archive[f'{name}_controller_{key}'] = getattr(syn['controller'], key)
        archive[f'{name}_storage'] = syn['storage']; archive[f'{name}_x'] = x
        archive[f'{name}_Z'] = syn['Z']; archive[f'{name}_S'] = syn['S']
        print('report', row, flush=True)
    pd.DataFrame(rows).to_csv(DATA/'rp_results.csv', index=False)
    np.savez(DATA/'rp_solutions.npz', **archive)


def gradient_consistency():
    study = rp.RPStudy()
    x = designs()['Compliance']
    r = study.evaluate(x, gradient=True)
    ref = study.spectral_gradient(r['modal'], r['synthesis'])
    err = float(la.norm(r['gradient']-ref)/la.norm(ref))
    checks = rp.gradient_check(study, x)
    out = {'sparse_fold_vs_spectral_relative_difference': err, 'finite_difference': checks}
    print(out, flush=True)
    (DATA/'rp_gradient_check.json').write_text(json.dumps(out, indent=2, default=float)+'\n')


def ablation():
    ref = rp.RPStudy(); x = designs()['RP certified']; rows = []
    for order in [8, 12, 16, 20]:
        cfg = replace(base.CFG, retained_modes=order)
        s = rp.RPStudy(cfg, ref.scales, ref.settings, ref.state_scale)
        r = s.evaluate(x)
        full = cs.closed_loop(cs.performance_plant(s.plant(r['modal'], True)), r['synthesis']['controller'])
        sc = cs.CertifiedStudy(cfg, ref.scales, ref.settings, ref.state_scale).evaluate(x)
        rows.append({'m': order, 'Gamma_rp': r['synthesis']['gamma'], 'Gamma_rp_mu': r['value'],
                     'full_norm': cs.hn(full)[0], 'delta': r['tail']['delta'], 'tail_norm': nc.tail_norm(s, r['modal']),
                     'floor_delta': r['tail']['floor_delta'], 'eta_mu': sc['synthesis']['eta'],
                     'scalar_margin': sc['margin'], 'delta_eta': r['tail']['delta']*sc['synthesis']['eta']})
        print('ablation', rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(DATA/'rp_ablation.csv', index=False)


def refine():
    coarse = rp.RPStudy(); x = designs()['RP certified']; rows = []
    for nx, ny in [(10, 4), (20, 8), (40, 16)]:
        cfg = replace(base.CFG, nelx=nx, nely=ny)
        fine = rp.RPStudy(cfg, coarse.scales, replace(coarse.settings, max_iterations=8), coarse.state_scale)
        if nx == 10:
            r = fine.evaluate(x, gradient=True)
        else:
            r, h = fine.optimize(cs.transfer_design(coarse, fine, x), True, f'rp_mesh_{nx}')
            h.to_csv(DATA/f'rp_mesh_{nx}_history.csv', index=False)
            np.savez(DATA/f'rp_mesh_{nx}_design.npz', x=r['x'])
            r = fine.evaluate(r['x'], gradient=True)
        mdl, syn, tail = r['modal'], r['synthesis'], r['tail']
        rows.append({'nelx': nx, 'nely': ny, 'dofs': len(fine.fe.free_dofs), 'm': fine.m,
                     'delta': tail['delta'], 'Gamma_rp_mu': syn['eta'], 'Gamma_rp': syn['gamma'],
                     'gap_normalized': mdl['gap'], 'mass_port_squared': tail['mass_port_squared'],
                     'mass': fine.fe.mass(r['x']), 'eigen_seconds': mdl['eigen_seconds'],
                     'sdp_seconds': syn['info']['sdp_seconds'], **r['timing']})
        print('mesh', rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(DATA/'rp_mesh_refinement.csv', index=False)


def sparse_and_clusters():
    coarse = cs.CertifiedStudy(); out = []
    paths = {10: DATA/'rp_certified_design.npz', 20: DATA/'rp_mesh_20_design.npz', 40: DATA/'rp_mesh_40_design.npz'}
    for nx, ny in [(10, 4), (20, 8), (40, 16)]:
        cfg = replace(base.CFG, nelx=nx, nely=ny)
        st = cs.CertifiedStudy(cfg, coarse.scales, cs.SET, coarse.state_scale)
        x = np.load(paths[nx])['x']
        mdl = st.modal(x); tail = st.tail(mdl)
        import scipy.sparse as sparse, scipy.sparse.linalg as spla
        K = sparse.csc_matrix(mdl['K']); M = sparse.csc_matrix(mdl['M'])
        t0 = time.perf_counter(); vals, vecs = spla.eigsh(K, k=st.m+1, M=M, sigma=0, which='LM'); te = time.perf_counter()-t0
        order = np.argsort(vals); smdl = dict(mdl); smdl['phi'] = vecs[:, order[:st.m]]; smdl['Lambda'] = np.diag(vals[order[:st.m]])
        t0 = time.perf_counter(); sur = rv.sparse_displacement_surrogate(st, smdl, 2*st.cfg.damping_ratio); ts = time.perf_counter()-t0
        lam = mdl['lam']; rel = {k: float((lam[k]-lam[k-1])/lam[k-1]) for k in range(16, 29)}
        kstar = max(rel, key=rel.get)
        row = {'nelx': nx, 'dofs': len(st.fe.free_dofs), 'delta_q_exact': tail['delta_q'],
               'delta_q_sparse': sur['delta_q_sparse'], 'ratio': sur['delta_q_sparse']/tail['delta_q'],
               'delta': tail['delta'], 'delta_sparse': float(np.hypot(sur['delta_q_sparse'], tail['delta_v'])),
               'dense_eigensolve_seconds': mdl['eigen_seconds'], 'sparse_eigensolve_seconds': te,
               'sparse_surrogate_seconds': ts,
               'eigen_relative_error': float(np.max(np.abs(np.sort(vals)-lam[:st.m+1])/lam[:st.m+1])),
               'relative_gap_m20': rel[20], 'window_best_m': kstar, 'window_best_gap': rel[kstar], 'relative_gaps': rel}
        out.append(row); print('sparse', row, flush=True)
    # Cluster alternatives on the finest RP design.
    x = np.load(paths[40])['x']; alt = []
    for m in [19, 20, 21]:
        cfg = replace(base.CFG, nelx=40, nely=16, retained_modes=m)
        st = rp.RPStudy(cfg, coarse.scales, cs.SET, coarse.state_scale)
        r = st.evaluate(x); lam = r['modal']['lam']
        alt.append({'m': m, 'relative_gap': float((lam[m]-lam[m-1])/lam[m-1]), 'gap_normalized': r['modal']['gap'],
                    'Gamma_rp': r['synthesis']['gamma'], 'Gamma_rp_mu': r['value'], 'delta': r['tail']['delta']})
        print('cluster', alt[-1], flush=True)
    (DATA/'rp_sparse_clusters.json').write_text(json.dumps({'sparse': out, 'clusters': alt}, indent=2)+'\n')


class NonCollocatedRP(rp.RPStudy, nc.NonCollocatedStudy):
    pass


def noncollocated():
    ref = nc.NonCollocatedStudy()
    x = base.initial_compliance_design(ref.old)[0]
    rows = []
    for m in [4, 6, 8, 20]:
        cfg = replace(base.CFG, retained_modes=m)
        s = nc.NonCollocatedStudy(cfg, ref.scales, cs.SET, ref.state_scale)
        r = s.evaluate(x)
        aug = cs.closed_loop(s.plant(r['modal']), r['synthesis']['controller'])
        zw, zv, rw, uv, _, _ = blocks(aug)
        t = r['tail']
        row = {'m': m, 'eta_mu': r['synthesis']['eta'], 'delta': t['delta'], 'delta_q': t['delta_q'], 'delta_v': t['delta_v'],
               'delta_eta': t['delta']*r['synthesis']['eta'], 'tail_norm': nc.tail_norm(s, r['modal']),
               'scalar_Uv_delta_v': uv*t['delta_v'],
               'scalar_block': zw+(zv*t['delta_v']+t['delta_q'])*rw/(1-t['delta_v']*uv),
               'scalar_full': cs.hn(cs.closed_loop(cs.performance_plant(s.plant(r['modal'], True)), r['synthesis']['controller']))[0]}
        q = NonCollocatedRP(cfg, ref.scales, cs.SET, ref.state_scale)
        try:
            rr = q.evaluate(x)
            aug = cs.closed_loop(q.plant(rr['modal']), rr['synthesis']['controller'])
            zw, zv, rw, uv, _, _ = blocks(aug)
            full = cs.closed_loop(cs.performance_plant(q.plant(rr['modal'], True)), rr['synthesis']['controller'])
            row.update(Gamma_rp=rr['synthesis']['gamma'], rp_full=cs.hn(full)[0],
                       rp_abscissa=float(cs.ct.poles(full).real.max()), rp_Uv_delta_v=uv*t['delta_v'],
                       rp_block=zw+(zv*t['delta_v']+t['delta_q'])*rw/(1-t['delta_v']*uv))
        except Exception as error:
            row.update(rp_error=str(error))
        rows.append(row); print('noncollocated', row, flush=True)
    pd.DataFrame(rows).to_csv(DATA/'rp_noncollocated.csv', index=False)


def noncollocated_damping():
    """Mid-span sensing at m=4 with the stiffness-proportional damping reduced by fixed factors."""
    ref = nc.NonCollocatedStudy()
    x = base.initial_compliance_design(ref.old)[0]
    cfg = replace(base.CFG, retained_modes=4)
    rows = []
    for factor in [1.0, 0.3, 0.1, 0.03, 0.01]:
        scales = replace(ref.scales, beta=ref.scales.beta*factor)
        s = nc.NonCollocatedStudy(cfg, scales, cs.SET, ref.state_scale)
        r = s.evaluate(x)
        t = r['tail']
        row = {'beta_factor': factor, 'eta_mu': r['synthesis']['eta'], 'delta': t['delta'], 'delta_q': t['delta_q'],
               'delta_v': t['delta_v'], 'delta_eta': t['delta']*r['synthesis']['eta'], 'tail_norm': nc.tail_norm(s, r['modal'])}
        q = NonCollocatedRP(cfg, scales, cs.SET, ref.state_scale)
        rr = q.evaluate(x)
        aug = cs.closed_loop(q.plant(rr['modal']), rr['synthesis']['controller'])
        zw, zv, rw, uv, _, _ = blocks(aug)
        full = cs.closed_loop(cs.performance_plant(q.plant(rr['modal'], True)), rr['synthesis']['controller'])
        row.update(Gamma_rp=rr['synthesis']['gamma'], rp_full=cs.hn(full)[0],
                   rp_abscissa=float(cs.ct.poles(full).real.max()), Uv=uv, rp_Uv_delta_v=uv*t['delta_v'],
                   rp_block=zw+(zv*t['delta_v']+t['delta_q'])*rw/(1-t['delta_v']*uv))
        rows.append(row); print('noncollocated damping', row, flush=True)
    pd.DataFrame(rows).to_csv(DATA/'rp_noncollocated_damping.csv', index=False)


def regularity():
    study = rp.RPStudy(); x = designs()['RP certified']
    out = {'sweep': [], 'mu': {}}
    r = study.evaluate(x, gradient=True)
    p = study.plant(r['modal'])
    for theta in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
        reg = rp_regularity(study, p, r, theta)
        out['sweep'].append(reg); print('rp regularity', reg, flush=True)
    g0 = r['gradient']
    for mu in [1e-5, 1e-6, 1e-7]:
        s = rp.RPStudy(study.cfg, study.scales, replace(cs.SET, mu=mu), study.state_scale)
        q = s.evaluate(x, gradient=True)
        full = cs.closed_loop(cs.performance_plant(s.plant(q['modal'], True)), q['synthesis']['controller'])
        out['mu'][f'{mu:.0e}'] = {'Gamma_rp_mu': q['value'], 'Gamma_rp': q['synthesis']['gamma'],
                                  'full_norm': cs.hn(full)[0], 'kkt': q['synthesis']['info']['relative_kkt_residual'],
                                  'gradient_relative_difference': float(la.norm(q['gradient']-g0)/la.norm(g0)),
                                  'gradient_cosine': float(q['gradient']@g0/(la.norm(q['gradient'])*la.norm(g0)))}
        print('rp mu', mu, out['mu'][f'{mu:.0e}'], flush=True)
    (DATA/'rp_regularity.json').write_text(json.dumps(out, indent=2)+'\n')


def rp_regularity(study, p, r, theta):
    """Single-threshold nondegeneracy test for the RP conic constraints."""
    syn = r['synthesis']; n = len(p[0]); x = syn['x']
    cin, cout = study.coefficients(r['tail']['delta'])
    Q = np.block([[x[0], np.eye(n)], [np.eye(n), x[1]]])-study.settings.coupling_margin*np.eye(2*n)
    S = -rp.lmi_rp(p, x, syn['gamma'], cin, cout)-study.settings.brl_margin*np.eye(2*n+7)
    iu = np.triu_indices(n); rows = []; kernels = []

    def pack(parts):
        out = []
        for k, a in enumerate(parts):
            if k < 2:
                z = a[iu].copy(); z[iu[0] != iu[1]] *= np.sqrt(2); out.append(z)
            else: out.append(np.ravel(a))
        return np.concatenate(out)
    for slack, which in [(Q, 'coupling'), (S, 'brl')]:
        se, U = la.eigh(rv.scaled(slack)); V = U[:, se < theta]; kernels.append(V.shape[1])
        for i in range(V.shape[1]):
            for j in range(i, V.shape[1]):
                seed = np.outer(V[:, i], V[:, j])
                if i != j: seed = (seed+seed.T)/np.sqrt(2)
                parts = (rp.dual_coefficients_rp(p, np.zeros_like(S), seed, 1.) if which == 'coupling'
                         else rp.dual_coefficients_rp(p, seed, np.zeros_like(Q), 1.))
                rows.append(pack(parts))
    sv = la.svdvals(np.asarray(rows))
    return {'theta': theta, 'kernels': kernels, 'rows': len(rows), 'sigma_ratio': float(sv[-1]/sv[0]),
            'sigma_min': float(sv[-1]), 'sigma_max': float(sv[0])}


def mu_search(mu):
    class Rejecting(rp.RPStudy):
        failures = 0

        def evaluate(self, x, gradient=False, certified=True, anchor=None):
            try:
                return super().evaluate(x, gradient, certified, anchor)
            except Exception as error:
                if anchor is None: raise
                Rejecting.failures += 1
                print('rejected candidate after solver failure:', error, flush=True)
                return {'x': np.asarray(x).copy(), 'modal': self.modal(x, anchor), 'margin': -1., 'value': np.inf,
                        'synthesis': {'info': {'pole_abscissa': 1., 'physical_brl_maximum_eigenvalue': 1.}}}
    study = Rejecting(settings=replace(cs.SET, mu=mu))
    r, h = study.optimize(designs()['Compliance'], True, f'rp_mu{mu:.0e}')
    h.to_csv(DATA/f'rp_mu{mu:.0e}_history.csv', index=False)
    ref = designs()['RP certified']
    full = cs.closed_loop(cs.performance_plant(study.plant(r['modal'], True)), r['synthesis']['controller'])
    row = {'mu': mu, 'Gamma_rp_mu': r['value'], 'Gamma_rp': r['synthesis']['gamma'], 'full_norm': cs.hn(full)[0],
           'iterations': len(h)-1, 'projected_residual': float(h['projected_residual'].iloc[-1]),
           'design_difference_max': float(np.max(np.abs(r['x']-ref))),
           'design_difference_rel': float(la.norm(r['x']-ref)/la.norm(ref)), 'solver_failures': Rejecting.failures}
    print('rp mu search', row, flush=True)
    (DATA/f'rp_mu{mu:.0e}_result.json').write_text(json.dumps(row, indent=2)+'\n')
    np.savez(DATA/f'rp_mu{mu:.0e}_design.npz', x=r['x'])


def main():
    ap = argparse.ArgumentParser()
    for flag in ['report', 'gradient', 'ablation', 'refine', 'sparse', 'noncollocated', 'noncollocated-damping', 'regularity', 'sparse-certificate']:
        ap.add_argument('--'+flag, action='store_true')
    ap.add_argument('--mu-search', type=float, default=None)
    a = ap.parse_args()
    if a.gradient: gradient_consistency()
    if a.noncollocated: noncollocated()
    if a.noncollocated_damping: noncollocated_damping()
    if a.report: report()
    if a.ablation: ablation()
    if a.regularity: regularity()
    if a.refine: refine()
    if a.sparse: sparse_and_clusters()
    if a.mu_search is not None: mu_search(a.mu_search)
    if a.sparse_certificate: sparse_certificate()


def sparse_certificate():
    """Robust-performance level when delta uses the sparse displacement surrogate."""
    coarse = rp.RPStudy(); rows = []
    paths = {10: DATA/'rp_certified_design.npz', 20: DATA/'rp_mesh_20_design.npz', 40: DATA/'rp_mesh_40_design.npz'}
    for nx, ny in [(10, 4), (20, 8), (40, 16)]:
        cfg = replace(base.CFG, nelx=nx, nely=ny)
        st = rp.RPStudy(cfg, coarse.scales, cs.SET, coarse.state_scale)
        x = np.load(paths[nx])['x']; mdl = st.modal(x); tail = st.tail(mdl)
        sur = rv.sparse_displacement_surrogate(st, mdl, 2*st.cfg.damping_ratio)
        dsp = float(np.hypot(sur['delta_q_sparse'], tail['delta_v']))
        p = st.plant(mdl)
        exact = st.rp.solve(p, *st.coefficients(tail['delta']))['gamma']
        sparse_level = st.rp.solve(p, *st.coefficients(dsp))['gamma']
        rows.append({'nelx': nx, 'delta': tail['delta'], 'delta_sparse': dsp, 'Gamma_rp': exact,
                     'Gamma_rp_sparse': sparse_level, 'relative_increase': sparse_level/exact-1})
        print('sparse certificate', rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(DATA/'rp_sparse_certificate.csv', index=False)


if __name__ == '__main__':
    main()
