"""Reusable finite-element, scaling, initialization and PGF plotting utilities."""
from __future__ import annotations
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Iterable
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'mplconfig'))
import control as ct
import matplotlib as mpl
mpl.use('pgf')
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / 'figures'
DATA_DIR = ROOT / 'data'
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

@dataclass(frozen=True)
class Config:
    seed: int = 20260820
    nelx: int = 10
    nely: int = 4
    length: float = 0.60
    height: float = 0.20
    thickness: float = 0.008
    young: float = 70.0e9
    poisson: float = 0.33
    material_density: float = 2700.0
    simp_p: float = 3.0
    e_floor: float = 1.0e-4
    rho_min: float = 0.05
    volume_fraction: float = 0.50
    filter_radius_elements: float = 1.55
    retained_modes: int = 20
    damping_ratio: float = 0.02
    force_reference: float = 1.0
    control_weight: float = 0.04


CFG = Config()

def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 2.0,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": False,
            "text.usetex": True,
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath,amssymb}",
        }
    )


def quad4_unit_matrices(dx: float, dy: float, nu: float, thickness: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return Q4 stiffness for E=1 and consistent mass for density=1."""
    dmat = np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]],
        dtype=float,
    ) / (1.0 - nu**2)
    coords = np.array([[0.0, 0.0], [dx, 0.0], [dx, dy], [0.0, dy]])
    ke = np.zeros((8, 8))
    me = np.zeros((8, 8))
    gp = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))
    for xi in gp:
        for eta in gp:
            n = 0.25 * np.array(
                [
                    (1.0 - xi) * (1.0 - eta),
                    (1.0 + xi) * (1.0 - eta),
                    (1.0 + xi) * (1.0 + eta),
                    (1.0 - xi) * (1.0 + eta),
                ]
            )
            dn_nat = 0.25 * np.array(
                [
                    [-(1.0 - eta), +(1.0 - eta), +(1.0 + eta), -(1.0 + eta)],
                    [-(1.0 - xi), -(1.0 + xi), +(1.0 + xi), +(1.0 - xi)],
                ]
            )
            jac = dn_nat @ coords
            det_j = float(np.linalg.det(jac))
            dn_xy = np.linalg.solve(jac, dn_nat)
            bmat = np.zeros((3, 8))
            nmat = np.zeros((2, 8))
            for a in range(4):
                bmat[0, 2 * a] = dn_xy[0, a]
                bmat[1, 2 * a + 1] = dn_xy[1, a]
                bmat[2, 2 * a] = dn_xy[1, a]
                bmat[2, 2 * a + 1] = dn_xy[0, a]
                nmat[0, 2 * a] = n[a]
                nmat[1, 2 * a + 1] = n[a]
            ke += thickness * det_j * (bmat.T @ dmat @ bmat)
            me += thickness * det_j * (nmat.T @ nmat)
    return ke, me


class CantileverFE:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ne = cfg.nelx * cfg.nely
        self.dx = cfg.length / cfg.nelx
        self.dy = cfg.height / cfg.nely
        self.nn = (cfg.nelx + 1) * (cfg.nely + 1)
        self.ndof = 2 * self.nn
        self.ke_unit, self.me_unit = quad4_unit_matrices(
            self.dx, self.dy, cfg.poisson, cfg.thickness
        )
        self.edofs = self._element_dofs()
        self.fixed_dofs = self._fixed_dofs()
        self.free_dofs = np.setdiff1d(np.arange(self.ndof), self.fixed_dofs)
        self.force_full = self._patch_force()
        self.force = self.force_full[self.free_dofs]
        self.filter_matrix = self._density_filter()
        self.nondesign = self._nondesign_elements()
        self.design = np.setdiff1d(np.arange(self.ne), self.nondesign)

    def node(self, ix: int, iy: int) -> int:
        return ix * (self.cfg.nely + 1) + iy

    def _element_dofs(self) -> np.ndarray:
        out = []
        for ex in range(self.cfg.nelx):
            for ey in range(self.cfg.nely):
                nodes = [
                    self.node(ex, ey),
                    self.node(ex + 1, ey),
                    self.node(ex + 1, ey + 1),
                    self.node(ex, ey + 1),
                ]
                dofs = []
                for no in nodes:
                    dofs.extend([2 * no, 2 * no + 1])
                out.append(dofs)
        return np.asarray(out, dtype=int)

    def _fixed_dofs(self) -> np.ndarray:
        nodes = [self.node(0, iy) for iy in range(self.cfg.nely + 1)]
        return np.asarray([d for no in nodes for d in (2 * no, 2 * no + 1)], dtype=int)

    def _patch_force(self) -> np.ndarray:
        # A three-node, design-independent vertical patch on the right boundary.
        mid = self.cfg.nely // 2
        ids = [max(0, mid - 1), mid, min(self.cfg.nely, mid + 1)]
        weights = np.array([0.25, 0.50, 0.25])
        f = np.zeros(self.ndof)
        for iy, weight in zip(ids, weights):
            f[2 * self.node(self.cfg.nelx, iy) + 1] -= weight
        return f

    def _nondesign_elements(self) -> np.ndarray:
        fixed = []
        # Solid support column.
        for ey in range(self.cfg.nely):
            fixed.append(ey)
        # Solid actuator/sensor patch in the final column.
        ex = self.cfg.nelx - 1
        for ey in (self.cfg.nely // 2 - 1, self.cfg.nely // 2):
            fixed.append(ex * self.cfg.nely + ey)
        return np.asarray(sorted(set(fixed)), dtype=int)

    def _density_filter(self) -> np.ndarray:
        centers = []
        for ex in range(self.cfg.nelx):
            for ey in range(self.cfg.nely):
                centers.append((ex + 0.5, ey + 0.5))
        centers = np.asarray(centers)
        h = np.zeros((self.ne, self.ne))
        r = self.cfg.filter_radius_elements
        for i in range(self.ne):
            dist = np.linalg.norm(centers - centers[i], axis=1)
            h[i, :] = np.maximum(0.0, r - dist)
        h /= h.sum(axis=1, keepdims=True)
        return h

    def full_design(self, x_free: np.ndarray) -> np.ndarray:
        x = np.ones(self.ne)
        x[self.design] = np.asarray(x_free)
        return x

    def physical_density(self, x_full: np.ndarray) -> np.ndarray:
        xp = self.filter_matrix @ np.asarray(x_full)
        xp[self.nondesign] = 1.0
        return np.clip(xp, self.cfg.rho_min, 1.0)

    def volume(self, x_free: np.ndarray) -> float:
        return float(np.mean(self.physical_density(self.full_design(x_free))))

    def mass(self, x_free: np.ndarray) -> float:
        xp = self.physical_density(self.full_design(x_free))
        area = self.dx * self.dy
        return float(self.cfg.material_density * self.cfg.thickness * area * xp.sum())

    def assemble(self, x_free: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.full_design(x_free)
        xp = self.physical_density(x)
        k = np.zeros((self.ndof, self.ndof))
        m = np.zeros((self.ndof, self.ndof))
        for e, ed in enumerate(self.edofs):
            efac = self.cfg.e_floor + (1.0 - self.cfg.e_floor) * xp[e] ** self.cfg.simp_p
            mfac = xp[e]
            k[np.ix_(ed, ed)] += self.cfg.young * efac * self.ke_unit
            m[np.ix_(ed, ed)] += self.cfg.material_density * mfac * self.me_unit
        return k[np.ix_(self.free_dofs, self.free_dofs)], m[np.ix_(self.free_dofs, self.free_dofs)], xp

    def directional_matrices(self, x_free: np.ndarray, direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xp = self.physical_density(self.full_design(x_free))
        dx = np.zeros(self.ne)
        dx[self.design] = direction
        dxp = self.filter_matrix @ dx
        dxp[self.nondesign] = 0.0
        kd = np.zeros((self.ndof, self.ndof))
        md = np.zeros_like(kd)
        for e, ed in enumerate(self.edofs):
            prefactor = self.cfg.young * (1.0 - self.cfg.e_floor) * self.cfg.simp_p * xp[e] ** (self.cfg.simp_p - 1.0) * dxp[e]
            kd[np.ix_(ed, ed)] += prefactor * self.ke_unit
            md[np.ix_(ed, ed)] += self.cfg.material_density * dxp[e] * self.me_unit
        free = np.ix_(self.free_dofs, self.free_dofs)
        return kd[free], md[free]


@dataclass
class Scales:
    alpha: float
    beta: float
    qref: float
    vref: float
    omega_scale: float


class Study:
    def __init__(self, fe: CantileverFE, cfg: Config):
        self.fe = fe
        self.cfg = cfg
        self.x_uniform = self._uniform_feasible_design()
        k, m, _ = fe.assemble(self.x_uniform)
        lam, phi = la.eigh(k, m, check_finite=False)
        omega = np.sqrt(np.maximum(lam, 0.0))
        m_idx = cfg.retained_modes - 1
        w_a, w_b = omega[0], omega[m_idx]
        zeta = cfg.damping_ratio
        alpha = 2.0 * zeta * w_a * w_b / (w_a + w_b)
        beta = 2.0 * zeta / (w_a + w_b)
        static_compliance = float(fe.force @ la.solve(k, fe.force, assume_a="pos"))
        qref = cfg.force_reference * static_compliance
        omega_scale = omega[m_idx]
        vref = omega_scale * qref
        self.scales = Scales(alpha=alpha, beta=beta, qref=qref, vref=vref, omega_scale=omega_scale)


    def _uniform_feasible_design(self) -> np.ndarray:
        lo = self.cfg.rho_min
        hi = 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            x = np.full(len(self.fe.design), mid)
            if self.fe.volume(x) < self.cfg.volume_fraction:
                lo = mid
            else:
                hi = mid
        return np.full(len(self.fe.design), 0.5 * (lo + hi))


def initial_compliance_design(study: Study, niter: int = 55) -> Tuple[np.ndarray, List[float]]:
    """Density-filtered OC baseline with analytic compliance sensitivities."""
    fe, cfg = study.fe, study.cfg
    x = study.x_uniform.copy()
    history = []
    hnorm = fe.filter_matrix
    target = cfg.volume_fraction
    for _ in range(niter):
        k, _, xp = fe.assemble(x)
        u = la.solve(k, fe.force, assume_a="pos", check_finite=False)
        comp = float(fe.force @ u)
        history.append(comp)
        dc_phys = np.zeros(fe.ne)
        for e, ed_full in enumerate(fe.edofs):
            loc = np.searchsorted(fe.free_dofs, ed_full)
            valid = (loc < len(fe.free_dofs)) & (fe.free_dofs[np.minimum(loc, len(fe.free_dofs) - 1)] == ed_full)
            if not np.all(valid):
                # Fixed-edge elements include constrained DOFs.  Form a full local
                # vector with zeros on constrained entries.
                ue = np.zeros(8)
                for j, gdof in enumerate(ed_full):
                    pos = np.searchsorted(fe.free_dofs, gdof)
                    if pos < len(fe.free_dofs) and fe.free_dofs[pos] == gdof:
                        ue[j] = u[pos]
            else:
                ue = u[loc]
            dc_phys[e] = -cfg.young * cfg.simp_p * (1.0 - cfg.e_floor) * xp[e] ** (cfg.simp_p - 1.0) * float(ue @ fe.ke_unit @ ue)
        dc_phys[fe.nondesign] = 0.0
        dc_full = hnorm.T @ dc_phys
        dv_phys = np.ones(fe.ne) / fe.ne
        dv_phys[fe.nondesign] = 0.0
        dv_full = hnorm.T @ dv_phys
        dc = dc_full[fe.design]
        dv = np.maximum(dv_full[fe.design], 1.0e-12)
        move = 0.12
        l1, l2 = 1.0e-12, 1.0e12
        for _ in range(80):
            lm = math.sqrt(l1 * l2)
            cand = np.clip(x * np.sqrt(np.maximum(1.0e-30, -dc / (lm * dv))), x - move, x + move)
            cand = np.clip(cand, cfg.rho_min, 1.0)
            if fe.volume(cand) > target:
                l1 = lm
            else:
                l2 = lm
        xnew = cand
        if np.max(np.abs(xnew - x)) < 2.0e-3:
            x = xnew
            break
        x = xnew
    return x, history


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png")
    plt.close(fig)


def label_panels(axes: Iterable[plt.Axes]) -> None:
    for ax, letter in zip(axes, ("(a)", "(b)")):
        ax.annotate(letter, xy=(0.5, 0.0), xycoords="axes fraction", xytext=(0, -48), textcoords="offset points", ha="center", va="top", fontsize=12, annotation_clip=False)


