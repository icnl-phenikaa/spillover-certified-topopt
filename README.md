# Certified closed-loop topology optimization — code

Python code that regenerates every numerical result, table, and figure of the manuscript
*Topology optimization of actively controlled elastic structures with certified modal spillover bounds*.

No data files are distributed. All results are computed from scratch by the scripts below.

## Requirements

- Python 3.11
- The pinned packages in `requirements.txt`:
  numpy 2.2.6, scipy 1.16.3, sympy 1.13.3, matplotlib 3.10.6, pandas 2.3.2,
  control 0.10.2, slycot 0.7.0, cvxpy 1.7.5, Mosek 11.1.2
- A MOSEK licence (free for academic use), placed at `~/mosek/mosek.lic` or pointed to by `MOSEKLM_LICENSE_FILE`
- A LaTeX installation with `pdflatex`, `latexmk`, PGF/TikZ, and Latin Modern fonts. The plots use Matplotlib's PGF backend.

## Run order

Each step reads only files written by earlier steps.

| Step | Command | Produces | Used in the manuscript |
|---|---|---|---|
| 1 | `python certified_studies.py` | Compliance start, scalar ($\eta_\mu$) design and history | Scalar design in Tables 2–4, Figs. 3–4 |
| 2 | `python certified_studies.py --nyquist` | Nyquist-constrained baseline | Nyquist design in Tables 2–4, Fig. 4 |
| 3 | `python rp_study.py --run --scaling 2` | Certified robust-performance design and history | Certified design, Figs. 2–3 |
| 4 | `python rp_pipeline.py --report --gradient --ablation --regularity --noncollocated --noncollocated-damping` | Certificates and exact norms, gradient check, truncation order, regularity test, non-collocated sensing | Tables 3–7; Fig. 5; Sec. 5.5 gradient check; Sec. 5.7 damping sweep |
| 5 | `python rp_pipeline.py --refine --sparse --sparse-certificate` | Refined meshes, cluster selection, sparse surrogate | Table 8, Table D1 |
| 6 | `python rp_pipeline.py --mu-search 1e-5` and `--mu-search 1e-7` | Repeated certified searches | Regularization sensitivity, Sec. 5.5 |
| 7 | `python diagnostic_studies.py --diagnostics --threshold-sweep --sparse-derivative` | Scalar-synthesis diagnostics, sparse-surrogate derivative check | Sec. 5.5 |
| 8 | `python mechanics_studies.py modal surcharge refine density split channels` | Modal data, peak decomposition, surcharge test | Tables 2 and 6, Sec. 5.3 |
| 9 | `python plot_certified.py` | Figs. 2–7 data plots | Figs. 2–7 |

Step 3 starts from the compliance design written by step 1, and steps 4–9 need the designs from steps 1–3. Steps 6 and 7 are independent of each other and of step 5, so they can run in parallel.

## Files

| File | Content |
|---|---|
| `finite_element.py` | Q4 plane-stress model, density filter, port scaling, compliance initialization, plot settings |
| `certified_studies.py` | Retained modal plant, full-order conic synthesis, controller recovery, structural adjoint, Rayleigh residual bounds, scalar and Nyquist searches |
| `rp_study.py` | Scaled robust-performance synthesis and the certified proximal search |
| `rp_pipeline.py` | All robust-performance studies of Section 5 |
| `noncollocated_study.py` | Mid-span sensor configuration |
| `diagnostic_studies.py` | Regularity and regularization diagnostics, sparse-surrogate check |
| `mechanics_studies.py` | Modal and mechanics quantities |
| `plot_certified.py` | Data figures |

## Notes

- **Resuming.** The searches save `*_checkpoint.npz` files and resume from them when a step is restarted. To start completely fresh, delete `data/`.
- **Agreement with the manuscript.** The computations are deterministic for a fixed environment. Different BLAS builds, solver versions, or thread counts can change the last digits and, through the accepted-step sequence, the final designs slightly.
- **Auxiliary output.** Step 1 also runs a fractional small-gain search and writes a few `conic_*` files. The manuscript does not report them; the design from that search is used only by the scalar diagnostics of step 7.

## Licence and citation

Licence: *to be added*. Please cite the manuscript when using this code.
