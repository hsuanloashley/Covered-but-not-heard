# (was a4_changepoint_its.py)
from common import *
import ruptures as rpt, statsmodels.api as sm
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
M=load(); res={}
EVENTS={'COVID onset (WHO PHEIC 2020-01-30)':'2020-01-30','Pelosi visit':'2022-08-02','Blinken Beijing visit':'2023-06-18','Taiwan election':'2024-01-13','Aid bill':'2024-04-23'}
M['week']=M.publish_date.dt.to_period('W-SUN').dt.start_time
# (a) article count: full corpus
cnt=M.groupby('week').size().rename('article_count')
A=M[M.annot_skip==False]
La=long_mentions(A,'actor',True); Ls=long_mentions(A,'source',True)
La['week']=La.publish_date.dt.to_period('W-SUN').dt.start_time; Ls['week']=Ls.publish_date.dt.to_period('W-SUN').dt.start_time
def wshare(L,c,minn=8):
    L=L[L.country.isin(['United States','China','Taiwan'])]
    g=L.groupby('week'); n=g.size(); s=g.country.apply(lambda x:(x==c).mean()*100)
    s[n<minn]=np.nan; return s,n
S=pd.DataFrame({'article_count':cnt})
S['taiwan_actor_share'],S['n_actor_mentions']=wshare(La,'Taiwan'); S['us_actor_share'],_=wshare(La,'United States')
S['taiwan_source_share'],S['n_source_mentions']=wshare(Ls,'Taiwan'); S['china_source_share'],_=wshare(Ls,'China')
S=S[(S.index>='2020-01-01')&(S.index<='2025-03-02')]
S['article_count']=S.article_count.fillna(0)
GAP=S.article_count<100
S['collection_gap']=GAP
res['collection_gap_weeks']=[str(d.date()) for d in S.index[GAP]]
S.loc[GAP,'article_count']=np.nan
S.to_csv(os.path.join(OUT,'AF_weekly_series.csv'))
res['weekly_n']=int(len(S)); res['weeks_with_missing_share']=int(S.taiwan_source_share.isna().sum())
SERIES=['article_count','taiwan_actor_share','taiwan_source_share','us_actor_share','china_source_share']
def detect(y,name):
    y=y.interpolate(limit_direction='both').values.astype(float)
    if name=='article_count': y=np.log1p(y)
    z=(y-y.mean())/y.std()
    n=len(z)
    # BIC-type penalty for mean-shift (l2) on standardized series: log(n) per change point
    pen_bic=np.log(n)
    algo=rpt.Pelt(model='l2',min_size=4,jump=1).fit(z)
    bk_bic=algo.predict(pen=pen_bic)[:-1]
    # elbow: scan penalties
    pens=np.linspace(0.5,30,30); ks=[len(algo.predict(pen=p))-1 for p in pens]
    costs=[rpt.Pelt(model='l2',min_size=4,jump=1).fit(z).predict(pen=p) for p in []]
    # elbow via max curvature on k vs pen (kneedle style)
    ks=np.array(ks); pens_u=pens
    # normalized curve of k against pen; choose pen where second difference largest
    dk=np.diff(ks); elbow_idx=int(np.argmax(np.abs(np.diff(dk))))+1 if len(dk)>2 else 0
    pen_elbow=float(pens_u[elbow_idx]); bk_elbow=algo.predict(pen=pen_elbow)[:-1]
    # rbf for robustness
    bk_rbf=rpt.Pelt(model='rbf',min_size=4,jump=1).fit(z).predict(pen=pen_bic)[:-1]
    return dict(pen_bic=pen_bic,bk_bic=bk_bic,pen_elbow=pen_elbow,bk_elbow=bk_elbow,bk_rbf=bk_rbf,pen_scan=list(zip(pens_u.round(2),ks)))
weeks=S.index
rows=[]; cps={}
for s in SERIES:
    d=detect(S[s],s); cps[s]=d
    for method,key in [('l2+BIC','bk_bic'),('l2+elbow','bk_elbow'),('rbf+BIC','bk_rbf')]:
        for b in d[key]:
            wk=weeks[b] if b<len(weeks) else weeks[-1]
            dist={ev:int(round((wk-pd.Timestamp(dt)).days/7)) for ev,dt in EVENTS.items()}
            nearest=min(dist,key=lambda k:abs(dist[k]))
            rows.append(dict(series=s,method=method,penalty=round(d['pen_bic'] if 'BIC' in method else d['pen_elbow'],2),changepoint_week=wk.date(),**{f'wks_from_{k.split(" (")[0]}':v for k,v in dist.items()},nearest_event=nearest,nearest_abs_weeks=abs(dist[nearest])))
CP=pd.DataFrame(rows); save_table(CP,'AF_changepoints',index=False,floatfmt='.2f')
# summary: events recovered (a CP within +-4 weeks) per series for primary method
prim=CP[CP.method=='l2+BIC']
rec={}
for ev in EVENTS:
    k=ev.split(' (')[0]
    rec[k]={s:bool((prim[prim.series==s][f'wks_from_{k}'].abs()<=4).any()) for s in SERIES}
REC=pd.DataFrame(rec).T; save_table(REC,'AF_events_recovered_within4wks')
res['R1']={'primary_method':'PELT, l2 cost on standardized weekly series, min_size=4, penalty=log(n) (BIC)','events_recovered_within_4_weeks':rec,
 'n_changepoints_primary':{s:len(cps[s]['bk_bic']) for s in SERIES},'changepoints_primary':{s:[str(weeks[b].date()) for b in cps[s]['bk_bic']] for s in SERIES}}
# ---- ITS around Pelosi ----
def its(y,pre0='2021-07-01',t0='2022-08-01',post1='2022-12-31',lags=4):
    d=S.loc[pre0:post1,[y]].copy(); d=d.interpolate(limit_direction='both'); d['t']=np.arange(len(d)); d['post']=(d.index>=pd.Timestamp(t0)).astype(int)
    t_0=d.t[d.post==1].min(); d['tpost']=np.where(d.post==1,d.t-t_0,0)
    X=sm.add_constant(d[['t','post','tpost']]); m=sm.OLS(d[y],X).fit(cov_type='HAC',cov_kwds={'maxlags':lags})
    ci=m.conf_int().loc['post']; pre_mean=d[d.post==0][y].mean(); post_mean=d[d.post==1][y].mean()
    return dict(outcome=y,n_weeks=len(d),n_pre=int((d.post==0).sum()),n_post=int((d.post==1).sum()),pre_mean=pre_mean,post_mean=post_mean,level_shift=m.params['post'],ci_low=ci[0],ci_high=ci[1],p=m.pvalues['post'],slope_pre=m.params['t'],slope_change=m.params['tpost'],slope_change_p=m.pvalues['tpost'],newey_west_lags=lags)
ITS=pd.DataFrame([its(y) for y in ['taiwan_source_share','taiwan_actor_share','us_actor_share','china_source_share']])
save_table(ITS,'R3_ITS_pelosi',index=False,floatfmt='.3f')
res['R3']=ITS.set_index('outcome').round(3).to_dict('index')
# ---- figure ----
fig,axes=plt.subplots(len(SERIES),1,figsize=(11,13),sharex=True)
for ax,s in zip(axes,SERIES):
    y=S[s]; ax.plot(weeks,y,color='#444',lw=1)
    if GAP.any(): ax.axvspan(S.index[GAP].min(),S.index[GAP].max(),color='#999',alpha=0.3)   # guard: no gap weeks -> no shading
    if s=='article_count': ax.set_yscale('log')
    for b in cps[s]['bk_bic']: ax.axvline(weeks[b],color='#d62728',lw=1.5,label='PELT change point' if b==cps[s]['bk_bic'][0] else None)
    for ev,dt in EVENTS.items(): ax.axvline(pd.Timestamp(dt),color='#1f77b4',ls=':',lw=1)
    ax.set_ylabel(s.replace('_',' '),fontsize=9)
for ev,dt in EVENTS.items(): axes[0].text(pd.Timestamp(dt),axes[0].get_ylim()[1],ev.split(' (')[0],rotation=90,fontsize=7,va='top',color='#1f77b4')
axes[0].legend(loc='lower left',fontsize=8); axes[0].set_title('Weekly series with PELT change points (red) and event dates (blue dotted)')
plt.tight_layout(); plt.savefig(os.path.join(OUT,'AF_fig_changepoints.png'),dpi=300)
save_json(res,'results_a4'); print(json.dumps(res,indent=1,default=str)[:5000]); print(CP[CP.method=='l2+BIC'].to_string())
