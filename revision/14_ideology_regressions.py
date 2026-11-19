# (was a5_ideology_regressions.py)
from common import *
from outlet_types import outlet_type
import statsmodels.formula.api as smf, statsmodels.api as sm, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
M=load(); res={}
L=M[(M.label_error==False)&M.allsides.notna()].copy()
L['outlet_type']=L.media_name.map(outlet_type)
for l in LABELS: L[l]=L[l].astype(int)
mp=L[['media_name','allsides','outlet_type']].drop_duplicates().merge(L.media_name.value_counts().rename('n_labeled_headlines'),left_on='media_name',right_index=True).sort_values('n_labeled_headlines',ascending=False)
mp.to_csv(os.path.join(OUT,'outlet_type_mapping.csv'),index=False)
P=L[L.in_paper_labels==True]
res['N']={'paper_labels_with_allsides':int(len(P)),'all_valid_labels_with_allsides':int(len(L)),'outlets_paper':int(P.media_name.nunique()),'outlets_all':int(L.media_name.nunique())}
def fit(df,y,spec):
    df=df.copy(); fe=None
    if spec=='old': f=f'{y} ~ bias_num'; kw={}
    elif spec=='cluster': f=f'{y} ~ bias_num'; kw=dict(cov_type='cluster',cov_kwds={'groups':df.media_name.astype('category').cat.codes.values})
    else:
        fe='yearmonth'
        # drop time cells without outcome variation (they contribute nothing to a FE logit and cause separation)
        keep=df.groupby(fe)[y].transform(lambda x: 0<x.sum()<len(x)); df=df[keep]
        if df.groupby(fe).size().min()<5 or len(df.groupby(fe))<8:  # fall back to quarterly FE if cells too thin
            fe='yq'; df['yq']=df.publish_date.dt.to_period('Q').astype(str); keep=df.groupby(fe)[y].transform(lambda x: 0<x.sum()<len(x)); df=df[keep]
        f=f'{y} ~ bias_num + C({fe})' + (' + C(outlet_type)' if spec=='fe_type_cluster' else '')
        kw=dict(cov_type='cluster',cov_kwds={'groups':df.media_name.astype('category').cat.codes.values})
    try:
        m=smf.glm(f,df,family=sm.families.Binomial()).fit(**kw) if spec!='old' else smf.logit(f,df).fit(disp=0)
        b=m.params['bias_num']; se=m.bse['bias_num']
        return dict(OR=np.exp(b),ci_low=np.exp(b-1.96*se),ci_high=np.exp(b+1.96*se),p=m.pvalues['bias_num'],N=int(m.nobs),fe=fe,converged=True)
    except Exception as e: return dict(OR=np.nan,ci_low=np.nan,ci_high=np.nan,p=np.nan,N=len(df),fe=fe,converged=False,error=str(e)[:80])
SPECS=[('old','Old: naive Logit'),('cluster','+ outlet-clustered SE'),('fe_cluster','+ year-month FE, clustered'),('fe_type_cluster','+ year-month FE + outlet type, clustered')]
rows=[]
for sample,df in [('paper4062',P),('all7616',L)]:
    for y in LABELS:
        for s,nm in SPECS:
            r=fit(df,y,s); rows.append(dict(smp=sample,label=y,spec=s,spec_name=nm,**r))
R=pd.DataFrame(rows); save_table(R,'R4_ideology_regressions_long',index=False)
# side-by-side wide tables
def wide(sample):
    d=R[R.smp==sample]; w=d.pivot(index='label',columns='spec',values=['OR','ci_low','ci_high','p','N'])
    cols=[]; out=pd.DataFrame(index=LABELS)
    for s,nm in SPECS:
        out[f'OR ({s})']=w['OR'][s].round(2); out[f'95% CI ({s})']=w['ci_low'][s].round(2).astype(str)+'-'+w['ci_high'][s].round(2).astype(str); out[f'p ({s})']=w['p'][s].round(3)
    out['N (old)']=w['N']['old']; out['N (new)']=w['N']['fe_type_cluster']; return out
save_table(wide('paper4062'),'R4_table_paper_sample_old_vs_new',floatfmt='.3f')
save_table(wide('all7616'),'R4_table_expanded_sample_old_vs_new',floatfmt='.3f')
# flips
def sig(r): return r.p<0.05
fl=[]
for sample in ['paper4062','all7616']:
    d=R[R.smp==sample].set_index(['label','spec'])
    for y in LABELS:
        o=d.loc[(y,'old')]; n=d.loc[(y,'fe_type_cluster')]; n2=d.loc[(y,'fe_cluster')]
        fl.append(dict(sample=sample,label=y,old_OR=round(o.OR,2),old_p=round(o.p,3),new_OR=round(n.OR,2),new_CI=f'{n.ci_low:.2f}-{n.ci_high:.2f}',new_p=round(n.p,3),new_OR_noType=round(n2.OR,2),new_p_noType=round(n2.p,3),
          old_sig=sig(o),new_sig=sig(n),direction_same=bool((o.OR>1)==(n.OR>1)),conclusion_flips=bool(sig(o)!=sig(n) or (o.OR>1)!=(n.OR>1)),negligible_new=bool(0.83<n.OR<1.2)))
FL=pd.DataFrame(fl); save_table(FL,'R5_conclusion_flags',index=False)
res['flags']=FL.to_dict('records')
# forest plots
for sample,fname,labs,title in [('paper4062','R4_fig5_forest_topics',TOPICS,'Figure 5: topic frames by AllSides rating (Left=1 ... Right=5)'),('paper4062','AG_fig_forest_stances',STANCES,'Appendix G: stance labels by AllSides rating')]:
    d=R[(R.smp==sample)&R.label.isin(labs)]
    fig,ax=plt.subplots(figsize=(8,0.6*len(labs)+2)); yy=np.arange(len(labs))
    for k,(s,nm,col) in enumerate([('old','Original (naive SE)','#999'),('fe_type_cluster','Year-month FE + outlet type, outlet-clustered SE','#d62728')]):
        dd=d[d.spec==s].set_index('label').reindex(labs); off=-0.15 if k==0 else 0.15
        ax.errorbar(dd.OR,yy+off,xerr=[dd.OR-dd.ci_low,dd.ci_high-dd.OR],fmt='o',color=col,label=nm,capsize=3)
    ax.axvline(1,color='k',lw=0.8); ax.set_yticks(yy); ax.set_yticklabels(labs); ax.set_xscale('log'); ax.set_xlabel('Odds ratio per one-step rightward shift (log scale)'); ax.set_title(title,fontsize=10); ax.legend(fontsize=8,loc='lower right'); ax.invert_yaxis()
    plt.tight_layout(); plt.savefig(os.path.join(OUT,fname+'.png'),dpi=300); plt.close()
save_json(res,'results_a5'); print(res['N']); print(FL.to_string())
