#!/usr/bin/env python3
"""Regenerate every manuscript plot with Matplotlib's PGF/LaTeX backend."""
import json
import numpy as np
import pandas as pd
import scipy.linalg as la
import certified_studies as cs

base=cs.base
plt=base.plt
COLORS={'Compliance':'#0072B2','Scalar':'#E69F00','Certified':'#009E73','Nyquist':'#CC79A7'}
# Archive names of the robust-performance report (rp_pipeline.py --report).
ARCHIVE={'Compliance':'Compliance','Scalar':'Augmented','Certified':'RP certified','Nyquist':'Nyquist'}


# Every legend sits inside its own panel; legend text is smaller than the axis labels.
LEG=dict(fontsize=7.5,frameon=True,framealpha=.92,edgecolor='0.8',handlelength=1.8,borderpad=.4,labelspacing=.3)


def square(*axes):
    for ax in axes: ax.set_box_aspect(1)


def save(fig,name):
    for ax in fig.axes: ax.grid(False)
    base.save_figure(fig,name)


def panels(axes):
    base.label_panels(axes)


def main():
    base.configure_plotting()
    cs.FIG.mkdir(exist_ok=True)
    # Only the assets written by this script (and the superseded block diagram) are replaced;
    # the TikZ system figure figures/fig7_system.* is compiled separately (README).
    stems={'fig1_topologies','fig2_full_model_performance','fig3_retained_order_ablation',
           'fig4_optimization_history','fig5_spectral_tail_bounds','fig6_mesh_and_timing','fig7_lft'}
    for path in cs.FIG.iterdir():
        if path.is_file() and path.stem in stems and path.suffix.lower() in {'.pdf','.png','.pgf','.svg','.eps'}:
            path.unlink()
    study=cs.CertifiedStudy()
    sols=np.load(cs.DATA/'rp_solutions.npz')
    metrics=pd.read_csv(cs.DATA/'rp_results.csv').set_index('design')
    modals={name:study.modal(sols[ARCHIVE[name]+'_x']) for name in COLORS}
    # Conic topologies use their robust-performance controllers.
    ctrls={name:cs.ct.ss(*(sols[ARCHIVE[name]+'_controller_'+key] for key in ['A','B','C','D']))
           for name in COLORS if name!='Nyquist'}
    nys=np.load(cs.DATA/'nyquist_design.npz')
    ctrls['Nyquist']=cs.ct.ss(*(nys[key] for key in ['A','B','C','D']))
    fig,axes=plt.subplots(1,2,figsize=(5.0,2.5),constrained_layout=True)
    for ax,name in zip(axes,['Compliance','Certified']):
        im=ax.imshow(modals[name]['rho'].reshape(study.cfg.nelx,study.cfg.nely).T,
                     origin='lower',extent=[0,.6,0,.2],cmap='viridis',vmin=0,vmax=1,
                     aspect='equal',interpolation='nearest')
        ax.set(xlabel=r'$x$ (m)',ylabel=r'$y$ (m)',title=name)
        ax.set_xticks([0,.3,.6]); ax.set_yticks([0,.1,.2])
    cb=fig.colorbar(im,ax=axes,shrink=.55,pad=.03,ticks=[0,.5,1]); cb.set_label('Density')
    panels(axes); save(fig,'fig1_topologies')

    allomega=np.concatenate([np.sqrt(m['lam']) for m in modals.values()])
    omega=np.unique(np.r_[np.geomspace(.05*allomega.min(),3*allomega.max(),4200),
                 (allomega[:,None]*np.array([.97,.99,.997,1,1.003,1.01,1.03])).ravel()])
    frequency=omega/(2*np.pi)
    fig,axes=plt.subplots(1,2,figsize=(5.0,2.9),constrained_layout=True)
    for name in COLORS:
        val=cs.frequency_values(study,modals[name],ctrls[name],omega)
        axes[0].plot(frequency,val,color=COLORS[name],label=name)
    axes[0].set_xlim(0,700)
    axes[0].set_ylim(0,17)
    certified=cs.frequency_values(study,modals['Certified'],ctrls['Certified'],omega)
    retained=cs.frequency_values(study,modals['Certified'],ctrls['Certified'],omega,True)
    axes[1].loglog(frequency,certified,color=COLORS['Certified'],label='Full model')
    axes[1].loglog(frequency,retained,'--',color='#0072B2',label='Retained model')
    axes[1].axhline(metrics.loc['RP certified','Gamma_rp'],color='#D55E00',label=r'$\Gamma_{\rm rp}$')
    axes[1].axhline(metrics.loc['RP certified','Gamma_block'],color='#CC79A7',ls=':',label=r'$\Gamma_{\rm blk}$')
    for ax in axes:
        ax.set(xlabel='Frequency (Hz)',ylabel=r'$\sigma_{\max}(T)$')
    axes[1].set_ylim(1e-4,40)
    axes[0].legend(loc='upper right',**LEG)
    axes[1].legend(loc='lower left',**LEG)
    square(*axes)
    panels(axes); save(fig,'fig2_full_model_performance')

    order=pd.read_csv(cs.DATA/'rp_ablation.csv').rename(columns={'scalar_margin':'margin'})
    fig,axes=plt.subplots(1,2,figsize=(5.0,2.9),constrained_layout=True)
    axes[0].semilogy(order.m,order.tail_norm,'o-',color='#0072B2',label=r'Exact $\|\Delta\|_\infty$')
    axes[0].semilogy(order.m,order.delta,'s-',color='#009E73',label='Rayleigh bound')
    axes[0].semilogy(order.m,order.floor_delta,'^--',color='#D55E00',label='Floor bound')
    axes[0].set(xlabel='Retained modes $m$',ylabel='Residual level')
    axes[0].set_ylim(2.5e-3,0.25)
    axes[0].legend(loc='upper right',**LEG)
    axes[1].plot(order.m,order.margin,'D-',color='#000000',ms=5,label=r'$1-\delta\eta_\mu$ (left)')
    axes[1].axhline(0,color='0.45',ls=':',lw=1.5,label='Small-gain boundary')
    axes[1].set_ylim(-.05,1.0)
    right=axes[1].twinx()
    right.plot(order.m,order.delta/order.tail_norm,'s--',color='#009E73',label='Rayleigh ratio (right)')
    right.plot(order.m,order.floor_delta/order.tail_norm,'^--',color='#CC79A7',label='Floor ratio (right)')
    right.set_ylim(0.4,3.0)
    axes[1].set(xlabel='Retained modes $m$',ylabel=r'$1-\delta\eta_\mu$')
    right.set_ylabel(r'Bound $/\,\|\Delta\|_\infty$')
    handles=axes[1].get_lines()+right.get_lines()
    right.legend(handles,[h.get_label() for h in handles],loc='lower right',bbox_to_anchor=(1.0,.06),**LEG)
    for ax in axes: ax.set_xticks(order.m)
    square(axes[0],axes[1],right)
    panels(axes); save(fig,'fig3_retained_order_ablation')

    fig,axes=plt.subplots(1,2,figsize=(5.0,2.9),constrained_layout=True)
    for name,path in [('Scalar','augmented_history.csv'),('Certified','rp_certified_history.csv')]:
        h=pd.read_csv(cs.DATA/path)
        axes[0].plot(h.iteration,h.objective/h.objective.iloc[0],color=COLORS[name],label=name)
        axes[1].semilogy(h.iteration,h.projected_residual,color=COLORS[name],label=name)
    axes[0].set(xlabel='Accepted iterate',ylabel='Objective / initial value')
    axes[1].set(xlabel='Accepted iterate',ylabel='Projected gradient residual')
    for ax in axes: ax.legend(loc='upper right',**LEG)
    square(*axes)
    panels(axes); save(fig,'fig4_optimization_history')

    mdl=modals['Certified']; lam=mdl['lam']; b=mdl['all_phi'].T@study.fe.force
    q,_,v,_=cs.rayleigh_weights(lam,study.scales.alpha,study.scales.beta)
    zeta=(study.scales.alpha+study.scales.beta*lam)/(2*np.sqrt(lam))
    z0=zeta.min(); kappa=1/(2*z0*np.sqrt(1-z0*z0)) if z0<1/np.sqrt(2) else 1.
    cutoff=[]
    for m in [4,8,12,16,20,30,40,60,80]:
        aq=study.cfg.force_reference/study.scales.qref; av=study.cfg.force_reference/study.scales.vref
        cutoff.append({'m':m,'omega_cutoff':np.sqrt(lam[m]),
          'rayleigh_q':aq*np.sum(b[m:]**2*q[m:]),'rayleigh_v':av*np.sum(b[m:]**2*v[m:]),
          'rayleigh_q_envelope':aq*q[m]*np.sum(b*b),'rayleigh_v_envelope':av*v[m]*np.sum(b*b),
          'floor_q':aq*kappa*np.sum(b[m:]**2/lam[m:]),
          'floor_v':av/(2*z0)*np.sum(b[m:]**2/np.sqrt(lam[m:])),
          'floor_q_envelope':aq*kappa*np.sum(b*b)/lam[m],
          'floor_v_envelope':av/(2*z0)*np.sum(b*b)/np.sqrt(lam[m])})
    cut=pd.DataFrame(cutoff); cut.to_csv(cs.DATA/'rayleigh_cutoffs.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(5.0,2.9),constrained_layout=True)
    for ax,channel in zip(axes,['q','v']):
        ax.loglog(cut.omega_cutoff,cut['rayleigh_'+channel],'o-',color='#009E73',label='Rayleigh trace')
        ax.loglog(cut.omega_cutoff,cut['rayleigh_'+channel+'_envelope'],'--',color='#0072B2',label='Rayleigh envelope')
        ax.loglog(cut.omega_cutoff,cut['floor_'+channel],'^-',color='#D55E00',label='Floor trace')
        ax.loglog(cut.omega_cutoff,cut['floor_'+channel+'_envelope'],':',color='#CC79A7',label='Floor envelope')
        ax.set(xlabel=r'$\omega_{m+1}$ (rad/s)',ylabel=r'$\delta_'+channel+'$')
        low=min(cut['rayleigh_'+channel].min(),cut['floor_'+channel].min())
        ax.set_ylim(low*1e-3,3*cut['floor_'+channel+'_envelope'].max())
        ax.legend(loc='lower left',**LEG)
    square(*axes)
    panels(axes); save(fig,'fig5_spectral_tail_bounds')

    mesh=pd.read_csv(cs.DATA/'rp_mesh_refinement.csv')
    fig,axes=plt.subplots(1,2,figsize=(5.0,2.9),constrained_layout=True)
    axes[0].semilogy(mesh.dofs,mesh.delta,'o-',color='#0072B2',label=r'$\delta$')
    axes[0].semilogy(mesh.dofs,mesh.gap_normalized,'s--',color='#009E73',label='Gap')
    axes[0].set(xlabel='Structural DOFs',ylabel='Dimensionless level')
    axes[0].legend(loc='center right',**LEG)
    names=['eigen_seconds','deflated_seconds','damping_solve_seconds','sdp_seconds','tail_gradient_seconds','contraction_seconds']
    labels=['Eigen','Deflated','Damping','SDP','Tail','Contraction']
    colors=['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7','#56B4E9']
    xpos=np.arange(len(mesh)); width=.115
    for j,(name,label,color) in enumerate(zip(names,labels,colors)):
        axes[1].bar(xpos+(j-2.5)*width,mesh[name],width,color=color,label=label)
    axes[1].set_yscale('log'); axes[1].set_xticks(xpos,[r'$10\times4$',r'$20\times8$',r'$40\times16$'])
    axes[1].set(xlabel='Mesh',ylabel='Time per evaluation (s)')
    axes[1].set_ylim(2e-4,3e4)
    axes[1].legend(loc='upper center',ncol=2,columnspacing=.7,handlelength=1.0,handletextpad=.4,**{k:v for k,v in LEG.items() if k!='handlelength'})
    square(*axes)
    panels(axes); save(fig,'fig6_mesh_and_timing')

    (cs.DATA/'plot_metadata.json').write_text(json.dumps({'backend':'pgf','texsystem':'pdflatex',
        'axes_font_points':12,'legend_font_points':7.5,'grid':False,'legends':'inside each panel',
        'panel_aspect':'square','panels':'letters below axes','colors':COLORS},indent=2)+'\n')


if __name__=='__main__': main()
