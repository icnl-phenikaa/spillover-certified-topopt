#!/usr/bin/env python3
"""Mechanical interpretation of the reported designs.

Modal data, port participation, static compliance, closed-loop modal damping,
the surcharge test of the block bound, frequencies under refinement, and the
density change field. Writes numerical data only; never writes manuscript text.
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import scipy.linalg as la

import certified_studies as cs
import finite_element as base
import rp_study as rp

DATA = cs.DATA
NAMES = {'Compliance': 'Compliance', 'Scalar': 'Augmented', 'Certified': 'RP certified', 'Nyquist': 'Nyquist'}


def ctrl_from(sol, key):
    return cs.ct.ss(*(sol[f'{key}_controller_{k}'] for k in 'ABCD'))


def dominant_pole(study, mdl, ctrl):
    """Closed-loop pole of the full model nearest to the open-loop first mode."""
    full = cs.closed_loop(cs.performance_plant(study.plant(mdl, True)), ctrl)
    poles = la.eigvals(full.A)*study.scales.omega_scale
    lam1 = mdl['lam'][0]; c1 = study.scales.alpha+study.scales.beta*lam1
    p_open = -c1/2+1j*np.sqrt(lam1-c1**2/4)
    p = poles[np.argmin(np.abs(poles-p_open))]
    return {'freq_hz': float(abs(p.imag)/(2*np.pi)), 'zeta': float(-p.real/abs(p))}


def modal_data():
    study = cs.CertifiedStudy(); sol = np.load(DATA/'rp_solutions.npz')
    a, b = study.scales.alpha, study.scales.beta
    rows = []
    for label, key in NAMES.items():
        x = sol[f'{key}_x']; mdl = study.modal(x); lam = mdl['lam']
        bi = mdl['all_phi'].T@study.fe.force
        w = np.sqrt(lam); zeta = (a+b*lam)/(2*w); m = study.m
        part = bi**2
        static = part/lam
        mobility = part/(a+b*lam)
        compliance = float(study.fe.force@la.solve(mdl['K'], study.fe.force))*study.cfg.force_reference/study.scales.qref
        rp_ctrl = ctrl_from(sol, key)
        s = study.evaluate(x)
        scalar_ctrl = s['synthesis']['controller']
        row = {'design': label, 'f1_hz': w[0]/(2*np.pi), 'f2_hz': w[1]/(2*np.pi), 'f3_hz': w[2]/(2*np.pi),
               'zeta1': zeta[0], 'zeta21': zeta[m], 'zeta_max': zeta.max(),
               'participation_mode1': part[0]/part.sum(), 'participation_retained': part[:m].sum()/part.sum(),
               'participation_discarded': part[m:].sum()/part.sum(),
               'static_flex_discarded': static[m:].sum()/static.sum(),
               'mobility_discarded': mobility[m:].sum()/mobility.sum(),
               'static_compliance': compliance, 'port_norm_squared': part.sum()}
        for tag, ctrl in [('rp', rp_ctrl), ('scalar', scalar_ctrl)]:
            d = dominant_pole(study, mdl, ctrl)
            row[f'{tag}_cl_freq_hz'] = d['freq_hz']; row[f'{tag}_cl_zeta'] = d['zeta']
        if label == 'Nyquist':
            nyq = np.load(DATA/'nyquist_design.npz')
            d = dominant_pole(study, mdl, cs.ct.ss(nyq['A'], nyq['B'], nyq['C'], nyq['D']))
            row['nyquist_cl_freq_hz'] = d['freq_hz']; row['nyquist_cl_zeta'] = d['zeta']
        rows.append(row); print('modal', {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(DATA/'mechanics_modal.csv', index=False)


def surcharge_check():
    """Prop. 3.5(c) and Cor. 3.11 on the certified topology across retained orders."""
    ref = rp.RPStudy(); x = np.load(DATA/'rp_certified_design.npz')['x']; rows = []
    for order in [8, 12, 16, 20]:
        cfg = replace(base.CFG, retained_modes=order)
        s = rp.RPStudy(cfg, ref.scales, ref.settings, ref.state_scale)
        r = s.evaluate(x); syn = r['synthesis']; delta = r['tail']['delta']
        aug = cs.closed_loop(s.plant(r['modal']), syn['controller'])
        n11 = cs.hn(aug[:2, :2])[0]; n12 = cs.hn(aug[:2, 2:])[0]
        n21 = cs.hn(aug[2:, :2])[0]; n22 = cs.hn(aug[2:, 2:])[0]
        psi_star = np.sqrt(n12/n21)
        an_star = rp.rp_level(s, r, d=psi_star)
        bound = n11+delta*n12*n21/(1-delta*n22)
        full = cs.closed_loop(cs.performance_plant(s.plant(r['modal'], True)), syn['controller'])
        rows.append({'m': order, 'delta': delta, 'n11': n11, 'n12': n12, 'n21': n21, 'n22': n22,
                     'psi_star': psi_star, 'Gamma_an_psistar': an_star, 'Gamma_an_psi2': rp.rp_level(s, r),
                     'Gamma_rp': syn['gamma'], 'block_2x2': bound, 'surcharge': an_star-n11,
                     'surcharge_bound': delta*n12*n21/(1-delta*n22), 'full_norm': cs.hn(full)[0]})
        print('surcharge', rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(DATA/'mechanics_surcharge.csv', index=False)


def design_split():
    """Split each certified level into retained performance and residual surcharge."""
    study = rp.RPStudy(); sol = np.load(DATA/'rp_solutions.npz')
    res = pd.read_csv(DATA/'rp_results.csv').set_index('design'); rows = []
    for label, key in NAMES.items():
        mdl = study.modal(sol[f'{key}_x']); ctrl = ctrl_from(sol, key)
        aug = cs.closed_loop(study.plant(mdl), ctrl)
        n11 = cs.hn(aug[:2, :2])[0]
        rows.append({'design': label, 'n11': n11, 'Gamma_rp': res.loc[key, 'Gamma_rp'],
                     'surcharge': res.loc[key, 'Gamma_rp']-n11, 'full_norm': res.loc[key, 'full_norm'],
                     'delta': res.loc[key, 'delta']})
        print('split', rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(DATA/'mechanics_split.csv', index=False)


def resonance_channels():
    """Closed-loop peak at the first bending resonance split into its channels."""
    study = cs.CertifiedStudy(); sol = np.load(DATA/'rp_solutions.npz'); rows = []
    for label, key in NAMES.items():
        x = sol[f'{key}_x']; mdl = study.modal(x); lam = mdl['lam']
        b = mdl['all_phi'].T@study.fe.force; w1 = np.sqrt(lam[0])
        om = np.unique(np.r_[np.geomspace(.3*w1, 3*w1, 20000), w1*np.linspace(.95, 1.05, 4001)]); s = 1j*om
        den = lam[:, None]+(study.scales.alpha+study.scales.beta*lam[:, None])*s[None, :]+s[None, :]**2
        mech = np.sum(b[:, None]**2/den, axis=0)
        q = study.cfg.force_reference/study.scales.qref*mech; v = study.cfg.force_reference/study.scales.vref*s*mech
        ctrls = {'rp': ctrl_from(sol, key), 'scalar': study.evaluate(x)['synthesis']['controller']}
        row = {'design': label, 'open_loop_peak': float(np.abs(q).max()),
               'modal_flexibility_1': float(b[0]**2/lam[0]*study.cfg.force_reference/study.scales.qref)}
        for tag, ctrl in ctrls.items():
            k = np.array([np.asarray(cs.ct.evalfr(ctrl, z/study.scales.omega_scale)).item() for z in s]); cl = 1/(1-k*v)
            dz, nz = q*cl, q*k*cl
            du, nu = study.cfg.control_weight*k*v*cl, study.cfg.control_weight*k*cl
            mat = np.stack([np.stack([dz, nz], -1), np.stack([du, nu], -1)], -2)
            sv = np.linalg.svd(mat, compute_uv=False)[:, 0]; i = int(np.argmax(sv))
            row.update({f'{tag}_peak': float(sv[i]), f'{tag}_peak_hz': float(om[i]/(2*np.pi)),
                        f'{tag}_dz': float(abs(dz[i])), f'{tag}_nz': float(abs(nz[i])),
                        f'{tag}_effort': float(max(abs(du[i]), abs(nu[i]))), f'{tag}_gain': float(abs(k[i]))})
        rows.append(row); print('channels', row, flush=True)
    pd.DataFrame(rows).to_csv(DATA/'mechanics_channels.csv', index=False)


def refinement_frequencies():
    coarse = cs.CertifiedStudy(); rows = []
    xc = np.load(DATA/'rp_certified_design.npz')['x']
    paths = {20: DATA/'rp_mesh_20_design.npz', 40: DATA/'rp_mesh_40_design.npz'}
    for nx, ny in [(10, 4), (20, 8), (40, 16)]:
        cfg = replace(base.CFG, nelx=nx, nely=ny)
        st = cs.CertifiedStudy(cfg, coarse.scales, cs.SET, coarse.state_scale)
        transferred = xc if nx == 10 else cs.transfer_design(coarse, st, xc)
        for tag, x in [('transferred', transferred), ('optimized', xc if nx == 10 else np.load(paths[nx])['x'])]:
            mdl = st.modal(x); f = np.sqrt(mdl['lam'][:3])/(2*np.pi)
            comp = float(st.fe.force@la.solve(mdl['K'], st.fe.force))*st.cfg.force_reference/st.scales.qref
            rows.append({'nelx': nx, 'design': tag, 'f1_hz': f[0], 'f2_hz': f[1], 'f3_hz': f[2],
                         'static_compliance': comp, 'delta': st.tail(mdl)['delta']})
            print('refine', rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(DATA/'mechanics_refinement.csv', index=False)


def density_change():
    study = cs.CertifiedStudy(); sol = np.load(DATA/'rp_solutions.npz')
    rc = study.modal(sol['Compliance_x'])['rho'].reshape(study.cfg.nelx, study.cfg.nely)
    rr = study.modal(sol['RP certified_x'])['rho'].reshape(study.cfg.nelx, study.cfg.nely)
    d = rr-rc
    out = {'compliance': rc.T[::-1].round(3).tolist(), 'certified': rr.T[::-1].round(3).tolist(),
           'difference_top_row_first': d.T[::-1].round(4).tolist(),
           'outer_rows_change': float(d[:, [0, -1]].sum()), 'inner_rows_change': float(d[:, 1:-1].sum()),
           'column_change': d.sum(axis=1).round(4).tolist()}
    print(json.dumps(out, indent=1))
    (DATA/'mechanics_density_change.json').write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    import sys
    tasks = {'modal': modal_data, 'surcharge': surcharge_check, 'refine': refinement_frequencies,
             'density': density_change, 'split': design_split, 'channels': resonance_channels}
    for t in sys.argv[1:] or list(tasks):
        tasks[t]()
