# (was a1_corpus_subset.py) - run after 09_build_master.py; adds `substantive` columns to master.pkl
from common import *
M=load()
res={}
# ---- Where does "Taiwan" appear? (articles with scraped text) ----
T=M[M.text.notna()].copy()
def lead(text,k=2):
    s=re.split(r'(?<=[.!?])\s+',str(text).strip())
    return ' '.join(s[:k])
pat=re.compile(r'taiwan|taipei|formosa',re.I)
T['tw_head']=T.title.fillna('').str.contains(pat)
T['tw_lead']=T.text.apply(lead).str.contains(pat)
T['tw_body']=T.text.str.contains(pat)
T['loc']=np.select([T.tw_head, T.tw_lead, T.tw_body],['headline','lead (not headline)','body only'],'not found in scraped text')
loc=T['loc'].value_counts(); locp=(100*loc/len(T)).round(1)
tab=pd.DataFrame({'n':loc,'share_pct':locp}); save_table(tab,'M1_taiwan_location')
T['substantive']=T.tw_head|T.tw_lead
full_head=M.title.fillna('').str.contains(pat)
res['M1']={'n_full_corpus':int(len(M)),'n_with_scraped_text':int(len(T)),
 'headline_share_full_corpus_pct':round(100*full_head.mean(),1),'n_headline_full_corpus':int(full_head.sum()),
 'location_counts_textsubset':tab.to_dict(),
 'n_substantive_textsubset':int(T.substantive.sum()),'substantive_share_textsubset_pct':round(100*T.substantive.mean(),1),
 'n_substantive_full_corpus_headline_or_lead':int(full_head.sum() + (T.tw_lead & ~T.tw_head).sum()),
 'gpt_skip_rate_pct_among_annotated':round(100*M[M.annot_attempted].annot_skip.mean(),1),
 'n_annotated_attempted':int(M.annot_attempted.sum()),'n_annotated_nonskip':int((M.annot_skip==False).sum())}
# cross-tab GPT-skip vs substantive
ct=pd.crosstab(T.loc[T.annot_attempted,'substantive'],T.loc[T.annot_attempted,'annot_skip'],margins=True); save_table(ct,'M1_substantive_vs_gptskip')
res['M1']['gpt_skip_rate_substantive_pct']=round(100*T[T.annot_attempted&T.substantive].annot_skip.mean(),1)
res['M1']['gpt_skip_rate_nonsubstantive_pct']=round(100*T[T.annot_attempted&~T.substantive].annot_skip.mean(),1)
M=M.merge(T[['id','substantive','tw_head','tw_lead','loc']],on='id',how='left')
M.to_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)),'master.pkl'))

# ---- Actor / source shares: full annotated vs substantive subset ----
A=M[(M.annot_skip==False)].copy()
A_sub=A[A.substantive==True]
res['AE']={'n_articles_full':int(len(A)),'n_articles_substantive':int(len(A_sub))}
rows=[]
for ext,lab in [(False,'paper dictionary'),(True,'extended mapping')]:
    for kind in ['actor','source']:
        for nm,D in [('full',A),('substantive',A_sub)]:
            L=long_mentions(D,kind,ext); sh,n=country_shares(L)
            rows.append({'mapping':lab,'kind':kind,'sample':nm,'n_mentions':n,**sh.to_dict()})
            ws=country_shares(L,'window').reindex(WINDOWS); ws.insert(0,'sample',nm); ws.insert(0,'kind',kind); ws.insert(0,'mapping',lab)
            save_table(ws,f'AE_windows_{kind}_{nm}_{"ext" if ext else "dict"}')
            res['AE'][f'{kind}_{nm}_{"ext" if ext else "dict"}_windows']=ws.drop(columns=['mapping','kind','sample']).to_dict()
ov=pd.DataFrame(rows); save_table(ov,'AE_overall_shares',index=False)
res['AE']['overall']=ov.to_dict('records')
# checks
def check(ws):
    order=all(ws.loc[w,'United States']>ws.loc[w,'China']>ws.loc[w,'Taiwan'] for w in WINDOWS)
    tw=ws['Taiwan']; u=(tw['COVID']>tw['Pelosi']) and (tw['Post-election']>tw['Pelosi']) and tw['Pelosi']==tw.min()
    return order,u
flags={}
for ext in ['dict','ext']:
    for nm in ['full','substantive']:
        ws=pd.read_csv(os.path.join(OUT,f'AE_windows_source_{nm}_{ext}.csv'),index_col=0).reindex(WINDOWS)
        wa=pd.read_csv(os.path.join(OUT,f'AE_windows_actor_{nm}_{ext}.csv'),index_col=0).reindex(WINDOWS)
        o,_=check(ws); _,ua=check(wa); _,us=check(ws)
        cn_gt_tw_pelosi=ws.loc['Pelosi','China']>ws.loc['Pelosi','Taiwan']
        flags[f'{nm}_{ext}']={'source_order_US>CN>TW_all_windows':bool(o),'taiwan_actor_share_U_shape(min at Pelosi)':bool(ua),'taiwan_source_share_min_at_Pelosi':bool(us),'china_sources>taiwan_sources_in_Pelosi':bool(cn_gt_tw_pelosi),'taiwan_source_max_pct':float(ws.Taiwan.max())}
res['AE']['flags']=flags
# comparison table: full vs substantive, window-level Taiwan actor & source share (extended)
comp=pd.DataFrame({
 'TW actor % (full)':pd.read_csv(os.path.join(OUT,'AE_windows_actor_full_ext.csv'),index_col=0).Taiwan,
 'TW actor % (subst.)':pd.read_csv(os.path.join(OUT,'AE_windows_actor_substantive_ext.csv'),index_col=0).Taiwan,
 'US actor % (full)':pd.read_csv(os.path.join(OUT,'AE_windows_actor_full_ext.csv'),index_col=0)['United States'],
 'US actor % (subst.)':pd.read_csv(os.path.join(OUT,'AE_windows_actor_substantive_ext.csv'),index_col=0)['United States'],
 'TW source % (full)':pd.read_csv(os.path.join(OUT,'AE_windows_source_full_ext.csv'),index_col=0).Taiwan,
 'TW source % (subst.)':pd.read_csv(os.path.join(OUT,'AE_windows_source_substantive_ext.csv'),index_col=0).Taiwan,
 'CN source % (full)':pd.read_csv(os.path.join(OUT,'AE_windows_source_full_ext.csv'),index_col=0).China,
 'CN source % (subst.)':pd.read_csv(os.path.join(OUT,'AE_windows_source_substantive_ext.csv'),index_col=0).China,
}).reindex(WINDOWS); save_table(comp,'AE_comparison_full_vs_substantive',floatfmt='.1f')
# figure
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(11,4),sharey=True)
for i,kind in enumerate(['actor','source']):
    for nm,ls in [('full','-'),('substantive','--')]:
        ws=pd.read_csv(os.path.join(OUT,f'AE_windows_{kind}_{nm}_ext.csv'),index_col=0).reindex(WINDOWS)
        for c,col in [('United States','#1f77b4'),('China','#d62728'),('Taiwan','#2ca02c')]:
            ax[i].plot(WINDOWS,ws[c],ls,color=col,marker='o',label=f'{c} ({nm})')
    ax[i].set_title(f'{kind.capitalize()} share by country'); ax[i].tick_params(axis='x',rotation=30); ax[i].set_ylabel('% of US/China/Taiwan mentions')
ax[1].legend(fontsize=7,ncol=2); plt.tight_layout(); plt.savefig(os.path.join(OUT,'AE_fig_full_vs_substantive.png'),dpi=300)

# ---- M2 neutral stance share ----
Lb=M[M.label_error==False].copy()
Lb['no_stance']=(Lb[STANCES].sum(axis=1)==0)
Lp=Lb[Lb.in_paper_labels==True]
m2=pd.DataFrame({'all_valid_labels_n':Lb.groupby('window').size(),'all_valid_no_stance_pct':(100*Lb.groupby('window').no_stance.mean()).round(1),
   'paper4062_n':Lp.groupby('window').size(),'paper4062_no_stance_pct':(100*Lp.groupby('window').no_stance.mean()).round(1)}).reindex(WINDOWS)
m2.loc['Overall']=[len(Lb),round(100*Lb.no_stance.mean(),1),len(Lp),round(100*Lp.no_stance.mean(),1)]
save_table(m2,'M2_neutral_share',floatfmt='.1f')
res['M2']={'overall_pct_valid7616':round(100*Lb.no_stance.mean(),1),'overall_pct_paper4062':round(100*Lp.no_stance.mean(),1),'by_window':m2.to_dict()}
res['labels']={'attempted':int(M.label_attempted.sum()),'valid':int((M.label_error==False).sum()),'paper':int((M.in_paper_labels==True).sum()),'api_errors':int((M.label_error==True).sum())}
save_json(res,'results_a1')
print(json.dumps(res,indent=1,default=float)[:6000])
