#!/usr/bin/env python3
"""Non-collocated benchmark: tip force/performance patch, mid-span velocity sensor.

The disturbance, actuator and displacement performance share the tip patch.
The velocity measurement averages a separate fixed solid patch at mid-span.
The residual velocity channel is therefore non-collocated, so the velocity
feedback term ||U_v|| delta_v is not structurally small.

This module writes numerical data only. It never writes manuscript text.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

import certified_studies as cs
import finite_element as base

DATA = cs.DATA
SENSOR_X = (.30, .36)
SENSOR_Y = (.05, .15)


class SensorPatchFE(cs.PhysicalPatchFE):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sensor_full = self._sensor_port()
        self.sensor = self.sensor_full[self.free_dofs]

    def _in_sensor(self, ex, ey):
        x, y = (ex+.5)*self.dx, (ey+.5)*self.dy
        return (SENSOR_X[0]-1e-12 <= x <= SENSOR_X[1]+1e-12
                and SENSOR_Y[0]-1e-12 <= y <= SENSOR_Y[1]+1e-12)

    def _sensor_port(self):
        out = np.zeros(self.ndof)
        for ex in range(self.cfg.nelx):
            for ey in range(self.cfg.nely):
                if self._in_sensor(ex, ey):
                    for ix, iy in [(ex, ey), (ex+1, ey), (ex+1, ey+1), (ex, ey+1)]:
                        out[2*self.node(ix, iy)+1] -= self.dx*self.dy/4
        return out/np.sum(np.abs(out))

    def _nondesign_elements(self):
        base_set = set(super()._nondesign_elements().tolist())
        extra = {ex*self.cfg.nely+ey for ex in range(self.cfg.nelx)
                 for ey in range(self.cfg.nely) if self._in_sensor(ex, ey)}
        return np.array(sorted(base_set | extra), dtype=int)


class NonCollocatedStudy(cs.CertifiedStudy):
    def __init__(self, cfg=base.CFG, scales=None, settings=cs.SET, state_scale=None):
        self.cfg, self.settings = cfg, settings
        self.fe = SensorPatchFE(cfg)
        self.old = base.Study(self.fe, cfg)
        self.scales = scales or self.old.scales
        self.old.scales = self.scales
        self.m = cfg.retained_modes
        self.synthesis = cs.ConicSynthesis(2*self.m, settings)
        self.state_scale = state_scale
        self.volume_gradient = (np.ones(self.fe.ne)@self.physical_jacobian())/self.fe.ne
        xlo = np.full(len(self.fe.design), cfg.rho_min)
        self.volume_offset = self.fe.volume(xlo)-self.volume_gradient@xlo
        if state_scale is None:
            mdl = self.modal(self.old.x_uniform)
            self.state_scale = 1.
            p = self.plant(mdl)
            self.state_scale = float(np.sqrt(la.norm(np.vstack([p[3], p[4]]))/la.norm(p[2])))

    def plant(self, mdl, full=False):
        a, b1, b2, c1, c2, d11, d12, d21 = super().plant(mdl, full)
        phi = mdl['all_phi'] if full else mdl['phi']
        nm = phi.shape[1]
        s = phi.T@self.fe.sensor
        c2 = np.r_[np.zeros(nm), s/self.state_scale][None, :]
        return a, b1, b2, c1, c2, d11, d12, d21

    def _damping_solves(self, mdl):
        if '_c_solution' not in mdl:
            start = time.perf_counter()
            damping = sparse.csc_matrix(self.scales.alpha*mdl['M']+self.scales.beta*mdl['K'])
            lu = spla.splu(damping)
            mdl['_c_solution'] = lu.solve(self.fe.force)
            mdl['_c_sensor'] = lu.solve(self.fe.sensor)
            mdl['_c_seconds'] = time.perf_counter()-start

    def tail(self, mdl, gradient=False):
        lam = mdl['lam']; phi = mdl['all_phi']
        b = phi.T@self.fe.force; s = phi.T@self.fe.sensor
        alpha, beta = self.scales.alpha, self.scales.beta
        q, qp, v, vp = cs.rayleigh_weights(lam, alpha, beta)
        q[:self.m] = 0; qp[:self.m] = 0
        aq = self.cfg.force_reference/self.scales.qref
        av = self.cfg.force_reference/self.scales.vref
        dq = aq*np.sum(b*b*q)
        self._damping_solves(mdl)
        R = la.inv(alpha*np.eye(self.m)+beta*mdl['Lambda'])
        br = mdl['phi'].T@self.fe.force; sr = mdl['phi'].T@self.fe.sensor
        a_v = float(self.fe.sensor@mdl['_c_sensor']-sr@R@sr)
        b_v = float(self.fe.force@mdl['_c_solution']-br@R@br)
        dv = av*np.sqrt(a_v*b_v)
        delta = float(np.hypot(dq, dv))
        zeta = (alpha+beta*lam[self.m:])/(2*np.sqrt(lam[self.m:]))
        zmin = np.min(zeta)
        kappa = 1/(2*zmin*np.sqrt(1-zmin*zmin)) if zmin < 1/np.sqrt(2) else 1.
        oldq = aq*kappa*np.sum(b[self.m:]**2/lam[self.m:])
        oldv = av/(2*zmin)*np.sqrt(np.sum(s[self.m:]**2/np.sqrt(lam[self.m:]))*np.sum(b[self.m:]**2/np.sqrt(lam[self.m:])))
        return {'delta': delta, 'delta_q': float(dq), 'delta_v': float(dv),
                'a_v': a_v, 'b_v': b_v, 'floor_delta': float(np.hypot(oldq, oldv)),
                'mass_port_squared': float(np.sum(b*b)), 'sensor_port_squared': float(np.sum(s*s))}

    def structural_gradient(self, mdl, syn, certified=True):
        p = self.plant(mdl)
        a, b1, b2, c1, c2 = cs.lmi_seeds(p, syn['x'], syn['Z'])
        m = self.m; w = self.scales.omega_scale; st = self.state_scale
        barL = cs.sym(-a[m:, :m]/w**2-self.scales.beta*a[m:, m:]/w)
        # Force columns and the tip displacement row share the force port.
        barb = (b1[m:, 0]+b2[m:, 0])*st*self.cfg.force_reference/(w*self.scales.vref)+c1[0, :m]/st
        # The measurement row uses the separate sensor port.
        bars = c2[0, m:]/st
        phi = mdl['phi']; lr = mdl['Lambda']
        force = self.fe.force[:, None]; sensor = self.fe.sensor[:, None]
        br = phi.T@force; sr = phi.T@sensor
        W = force@barb[None, :]+sensor@bars[None, :]
        tail = self.tail(mdl)
        eta = syn['eta']; margin = 1-tail['delta']*eta
        directK = np.zeros_like(mdl['K']); directM = np.zeros_like(mdl['M'])
        if certified:
            barL /= margin**2
            W /= margin**2
            av = self.cfg.force_reference/self.scales.vref
            coef = eta**2*tail['delta_v']/(tail['delta']*margin**2)*av/2
            ws = np.sqrt(tail['b_v']/tail['a_v']); wb = np.sqrt(tail['a_v']/tail['b_v'])
            R = la.inv(self.scales.alpha*np.eye(m)+self.scales.beta*lr)
            gproj = ws*sr@sr.T+wb*br@br.T
            barL += self.scales.beta*coef*R@gproj@R
            W -= 2*coef*(ws*sensor@sr.T+wb*force@br.T)@R
            cs_, cb_ = mdl['_c_sensor'], mdl['_c_solution']
            invseed = coef*(ws*np.outer(cs_, cs_)+wb*np.outer(cb_, cb_))
            directK = -self.scales.beta*invseed
            directM = -self.scales.alpha*invseed
        theta = cs.sym(lr@barL)+.5*cs.sym(phi.T@W)
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
        Kseed = cs.sym(phi@barL@phi.T-Xi@phi.T)+directK
        Mseed = cs.sym(-phi@theta@phi.T+Xi@lr@phi.T)+directM
        if certified:
            q, qp, _, _ = cs.rayleigh_weights(mdl['lam'], self.scales.alpha, self.scales.beta)
            q[:m] = 0; qp[:m] = 0
            ap = mdl['all_phi']; ab = ap.T@self.fe.force
            qcoef = eta**2*tail['delta_q']*self.cfg.force_reference/(tail['delta']*self.scales.qref*margin**2)
            rankseed = np.outer(ab, ab)
            F = cs.divided_difference(mdl['lam'], q, qp)
            G = cs.divided_difference(mdl['lam'], mdl['lam']*q, q+mdl['lam']*qp)
            Kseed += qcoef*cs.sym(ap@(F*rankseed)@ap.T)
            Mseed -= qcoef*cs.sym(ap@(G*rankseed)@ap.T)
        ke_seed = np.zeros(self.fe.ne); me_seed = np.zeros(self.fe.ne)
        global_k = np.zeros((self.fe.ndof, self.fe.ndof)); global_m = np.zeros_like(global_k)
        free = np.ix_(self.fe.free_dofs, self.fe.free_dofs)
        global_k[free], global_m[free] = Kseed, Mseed
        for e, ed in enumerate(self.fe.edofs):
            ke_seed[e] = np.sum(global_k[np.ix_(ed, ed)]*self.fe.ke_unit)*self.cfg.young*(1-self.cfg.e_floor)*self.cfg.simp_p*mdl['rho'][e]**(self.cfg.simp_p-1)
            me_seed[e] = np.sum(global_m[np.ix_(ed, ed)]*self.fe.me_unit)*self.cfg.material_density
        grad = self.physical_jacobian().T@(ke_seed+me_seed)
        return grad, {}


def tail_norm(study, mdl):
    nm = len(mdl['lam']); m = study.m
    idx = np.r_[np.arange(m, nm), np.arange(nm+m, 2*nm)]
    pf = study.plant(mdl, True)
    ts = cs.ct.ss(pf[0][np.ix_(idx, idx)], pf[2][idx, :], np.vstack([pf[3][0, idx], pf[4][0, idx]]), np.zeros((2, 1)))
    return cs.hn(ts)[0]


def summary(study, name, r):
    mdl, syn, tail = r['modal'], r['synthesis'], r['tail']
    p = study.plant(mdl)
    aug = cs.closed_loop(p, syn['controller'])
    achieved, _ = cs.hn(aug)
    full = cs.closed_loop(cs.performance_plant(study.plant(mdl, True)), syn['controller'])
    full_abscissa = float(cs.ct.poles(full).real.max())
    full_norm, _ = cs.hn(full)
    zw, _ = cs.hn(aug[:2, :2]); zv, _ = cs.hn(aug[:2, 3]); rw, _ = cs.hn(aug[2, :2]); uv, _ = cs.hn(aug[2, 3])
    fm = 1-tail['delta_v']*uv
    block = zw+(zv*tail['delta_v']+tail['delta_q'])*rw/fm
    lower = study.synthesis.lower_bound(p)
    tn = tail_norm(study, mdl)
    row = {'design': name, 'eta_mu': syn['eta'], 'gamma_mu': syn['gamma'], 'achieved_eta': achieved,
           'eta_lb': lower['dual'], 'delta': tail['delta'], 'delta_q': tail['delta_q'], 'delta_v': tail['delta_v'],
           'tail_norm': tn, 'floor_delta': tail['floor_delta'], 'small_gain_margin': r['margin'],
           'Gamma_mu': syn['eta']/r['margin'], 'Gamma_block': block, 'Uv': uv, 'Zv': zv, 'Zw': zw, 'Rw': rw,
           'Uv_delta_v': uv*tail['delta_v'], 'feedback_margin': fm, 'full_norm': full_norm,
           'full_pole_abscissa': full_abscissa, 'gap_normalized': mdl['gap'],
           'volume': study.fe.volume(r['x']), 'mass': study.fe.mass(r['x']),
           'physical_brl': syn['info']['physical_brl_maximum_eigenvalue'],
           'kkt': syn['info']['relative_kkt_residual']}
    print('final', row, flush=True)
    return row


def gradient_check(study, x):
    r = study.evaluate(x, gradient=True)
    rng = np.random.default_rng(20261003)
    out = []
    for index in range(2):
        d = rng.normal(size=len(x)); d[(x < .051) | (x > .999)] = 0
        v = study.volume_gradient.copy(); v[(x < .051) | (x > .999)] = 0
        d -= v*(v@d)/(v@v); d /= la.norm(d)
        h = 3e-4
        rp = study.evaluate(x+h*d); rm = study.evaluate(x-h*d)
        fd = (rp['value']-rm['value'])/(2*h); ad = r['gradient']@d
        eta_fd = (rp['synthesis']['eta']-rm['synthesis']['eta'])/(2*h)
        eta_ad = study.structural_gradient(r['modal'], r['synthesis'], False)[0]@d
        hd = 1e-5
        tp = study.tail(study.modal(x+hd*d))['delta_v']; tm = study.tail(study.modal(x-hd*d))['delta_v']
        out.append({'complete_fd': fd, 'complete_adjoint': ad,
                    'complete_relative_error': abs(fd-ad)/max(1, abs(fd), abs(ad)),
                    'retained_fd': eta_fd, 'retained_adjoint': eta_ad,
                    'retained_relative_error': abs(eta_fd-eta_ad)/max(1, abs(eta_fd), abs(eta_ad)),
                    'delta_v_fd': (tp-tm)/(2*hd)})
        print('noncollocated gradient check', out[-1], flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report', action='store_true')
    a = parser.parse_args()
    study = NonCollocatedStudy()
    print('scales', study.scales, 'state scale', study.state_scale, 'nondesign', len(study.fe.nondesign), flush=True)
    xcomp, _ = base.initial_compliance_design(study.old)
    if a.probe:
        r = study.evaluate(xcomp)
        summary(study, 'Compliance', r)
    if a.check:
        res = gradient_check(study, xcomp)
        (DATA/'noncollocated_gradient_check.json').write_text(json.dumps(res, indent=2)+'\n')
    if a.run:
        aug, haug = study.optimize(xcomp, False, 'noncollocated_augmented')
        haug.to_csv(DATA/'noncollocated_augmented_history.csv', index=False)
        cert, hcert = study.optimize(xcomp, True, 'noncollocated_certified')
        hcert.to_csv(DATA/'noncollocated_certified_history.csv', index=False)
        np.savez(DATA/'noncollocated_designs.npz', compliance=xcomp, augmented=aug['x'], certified=cert['x'])
    if a.report or a.run:
        saved = np.load(DATA/'noncollocated_designs.npz')
        rows = [summary(study, name, study.evaluate(saved[key]))
                for name, key in [('Compliance', 'compliance'), ('Augmented', 'augmented'), ('Certified', 'certified')]]
        pd.DataFrame(rows).to_csv(DATA/'noncollocated_results.csv', index=False)
        meta = {'sensor_patch': {'x': SENSOR_X, 'y': SENSOR_Y}, 'force_patch': {'x': [.54, .60], 'y': [.05, .15]},
                'scales': {k: float(v) for k, v in vars(study.scales).items()}, 'state_scale': study.state_scale,
                'settings': {k: v for k, v in vars(study.settings).items()} if hasattr(study.settings, '__dict__') else None}
        (DATA/'noncollocated_summary.json').write_text(json.dumps(meta, indent=2, default=float)+'\n')


if __name__ == '__main__':
    main()
