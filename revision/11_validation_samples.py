# (was a2_validation_samples.py)
from common import *
M=load(); rng=np.random.RandomState(20250822)
VAL=os.path.join(OUT,'validation'); os.makedirs(VAL,exist_ok=True)
def strat_sample(df,strata,n_total,min_per_cell):
    g=df.groupby(strata,observed=True).size()
    base=np.minimum(g,min_per_cell)
    rem=n_total-base.sum()
    extra=((g-base)/(g-base).sum()*rem)
    alloc=base+np.floor(extra).astype(int)
    frac=(extra-np.floor(extra)).sort_values(ascending=False)
    for key in frac.index:
        if alloc.sum()>=n_total: break
        if alloc[key]<g[key]: alloc[key]+=1
    alloc=np.minimum(alloc,g)
    parts=[]
    for key,k in alloc.items():
        if k<=0: continue
        sub=df[(df[strata]==pd.Series(key,index=strata)).all(axis=1)] if len(strata)>1 else df[df[strata[0]]==key]
        parts.append(sub.sample(n=int(k),random_state=rng))
    return pd.concat(parts), alloc
res={}
# ---- headlines ----
F=M[(M.label_error==False)&M.allsides.notna()].copy()
H,alloc=strat_sample(F,['allsides','window'],500,10)
H=H.sample(frac=1,random_state=rng).reset_index(drop=True); H['sample_id']=['H%04d'%i for i in range(1,len(H)+1)]
pub=H[['sample_id','title','media_name','publish_date','allsides','window']].copy()
for l in LABELS: pub['human_'+l]=''
pub.to_csv(os.path.join(VAL,'validation_headlines.csv'),index=False)
pub.drop(columns=['allsides','window']).to_csv(os.path.join(VAL,'validation_headlines_coder2.csv'),index=False)
H[['sample_id','id','title','media_name','publish_date','allsides','window','label_model']+LABELS].to_csv(os.path.join(VAL,'KEY_validation_headlines.csv'),index=False)
al=alloc.reset_index(); al.columns=['allsides','window','n']; al=al.pivot(index='allsides',columns='window',values='n').reindex(columns=WINDOWS)
save_table(al,'M3_headline_sample_allocation',floatfmt='.0f')
res['M3']={'n':int(len(H)),'frame_n':int(len(F)),'allocation':al.to_dict()}
# ---- articles ----
FA=M[(M.annot_skip==False)&M.text.notna()&M.allsides.notna()&(M.label_error==False)].copy()
FA['crisis_label']=FA.crisis.astype(int)
Ar,alloc2=strat_sample(FA,['allsides','window','crisis_label'],400,3)
Ar=Ar.sample(frac=1,random_state=rng).reset_index(drop=True); Ar['sample_id']=['A%04d'%i for i in range(1,len(Ar)+1)]
pubA=Ar[['sample_id','title','media_name','publish_date','text']].copy()
for c in ['human_actor_1','human_actor_2','human_actor_3','human_source_1','human_source_2','human_source_3','human_actor_countries','human_source_countries','notes']: pubA[c]=''
pubA.to_csv(os.path.join(VAL,'validation_articles.csv'),index=False)
pubA.to_csv(os.path.join(VAL,'validation_articles_coder2.csv'),index=False)
Ar[['sample_id','id','title','media_name','publish_date','allsides','window','crisis_label']+ACOLS+SCOLS].to_csv(os.path.join(VAL,'KEY_validation_articles.csv'),index=False)
al2=alloc2.reset_index(); al2.columns=['allsides','window','crisis','n']
save_table(al2,'M4_article_sample_allocation',index=False,floatfmt='.0f')
res['M4']={'n':int(len(Ar)),'frame_n':int(len(FA)),'n_cells_nonempty':int((alloc2>0).sum()),'n_cells_total':int(len(alloc2)),'crisis_yes':int(Ar.crisis_label.sum())}
save_json(res,'results_a2'); print(res)
