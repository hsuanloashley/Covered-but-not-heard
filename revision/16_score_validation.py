#!/usr/bin/env python3
"""Score human validation files against model keys.

Usage:
  python score_validation.py --headlines validation_headlines.csv [--headlines2 validation_headlines_coder2.csv]
                             --articles validation_articles.csv [--articles2 validation_articles_coder2.csv]
                             --outdir revision_outputs/validation_scores

Headline file: columns human_<label> in {0,1} for the 11 labels (blank = not coded, row skipped).
Article file: human_actor_1..3, human_source_1..3 (free text, entity names), and optionally
  human_actor_countries / human_source_countries as semicolon-separated country lists
  (e.g. "United States; Taiwan; Taiwan"). If the country columns are blank, entity names are
  classified with the same country_map.classify() rules used for the GPT-4o annotations.
Matching of entities (for Table D3 at the entity level) is fuzzy: normalised-string token overlap >= 0.6
or one name contained in the other. Country-level precision/recall compares the multiset of countries.
Outputs: table_D1_headline_prf.csv/.md, table_D2_headline_errors.csv, table_D3_actor_source_country_prf.csv/.md,
  table_D4_annotation_errors.csv, krippendorff_alpha.csv (if second coder supplied), summary.json
"""
import argparse, os, sys, re, json, collections
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from country_map import classify, strip_role
LABELS=['crisis','independence','unification','democratic_values','natural_disaster','semiconductors','cultural','pro_Taiwan','against_Taiwan','pro_China','against_China']
HERE=os.path.dirname(os.path.abspath(__file__))
KEYDIR=os.path.join(HERE,'..','revision_outputs','validation')

def prf(y,yhat):
    y=np.asarray(y).astype(int); yhat=np.asarray(yhat).astype(int)
    tp=((y==1)&(yhat==1)).sum(); fp=((y==0)&(yhat==1)).sum(); fn=((y==1)&(yhat==0)).sum(); tn=((y==0)&(yhat==0)).sum()
    p=tp/(tp+fp) if tp+fp else np.nan; r=tp/(tp+fn) if tp+fn else np.nan
    f=2*p*r/(p+r) if (p+r) and not np.isnan(p) and not np.isnan(r) else np.nan
    return dict(n=len(y),human_pos=int(y.sum()),model_pos=int(yhat.sum()),tp=int(tp),fp=int(fp),fn=int(fn),precision=p,recall=r,f1=f,accuracy=(tp+tn)/len(y))

def kripp(a,b):
    import krippendorff
    m=np.array([a,b],dtype=float)
    try: return krippendorff.alpha(reliability_data=m,level_of_measurement='nominal')
    except Exception as e: return np.nan

ALIAS={'chinese':'china','chinas':'china','beijing':'china','taiwanese':'taiwan','taiwans':'taiwan','taipei':'taiwan','american':'united states','america':'united states','us':'united states','usa':'united states','washington':'united states','prc':'china','pla':'pla'}
def norm_ent(x):
    x=strip_role(x); x=re.sub(r"[^a-z0-9 ]",'',x); x=re.sub(r'\b(the|of|mr|ms|dr|president|minister|ministry|ministrys|secretary|spokesperson|spokesman|official|officials|government|forces|navy|military|and|allied)\b',' ',x)
    x=' '.join(ALIAS.get(w,w) for w in x.split())
    return x
def ent_match(a,b):
    a,b=norm_ent(a),norm_ent(b)
    if not a or not b: return False
    if a==b or a in b or b in a: return True
    ta,tb=set(a.split()),set(b.split())
    return len(ta&tb)/max(1,min(len(ta),len(tb)))>=0.6

def score_headlines(args,outdir,summary):
    key=pd.read_csv(os.path.join(KEYDIR,'KEY_validation_headlines.csv'))
    h=pd.read_csv(args.headlines)
    d=key.merge(h[['sample_id']+['human_'+l for l in LABELS]],on='sample_id')
    coded=d.dropna(subset=['human_'+l for l in LABELS])
    rows=[]; errs=[]
    for l in LABELS:
        y=coded['human_'+l].astype(int); yhat=coded[l].astype(int)
        rows.append(dict(label=l,**prf(y,yhat)))
        wrong=coded[y!=yhat]
        for _,r in wrong.iterrows():
            errs.append(dict(label=l,sample_id=r.sample_id,headline=r.title,outlet=r.media_name,human=int(r['human_'+l]),model=int(r[l]),error_type='false positive' if r[l]==1 else 'false negative'))
    D1=pd.DataFrame(rows); D1.to_csv(os.path.join(outdir,'table_D1_headline_prf.csv'),index=False)
    open(os.path.join(outdir,'table_D1_headline_prf.md'),'w').write(D1.to_markdown(index=False,floatfmt='.3f'))
    E=pd.DataFrame(errs)
    if len(E):
        # highest impact: errors on rarer labels (small human_pos) and on stance labels first
        pos=D1.set_index('label').human_pos; E['impact']=E.label.map(lambda l: 1/(1+pos[l]))+E.label.isin(LABELS[7:])*0.5
        E.sort_values('impact',ascending=False).head(20).drop(columns='impact').to_csv(os.path.join(outdir,'table_D2_headline_errors.csv'),index=False)
    summary['headlines']={'n_coded':int(len(coded)),'macro_precision':float(D1.precision.mean()),'macro_recall':float(D1.recall.mean()),'macro_f1':float(D1.f1.mean()),
        'weakest_label':str(D1.sort_values('f1').label.iloc[0]) if D1.f1.notna().any() else None,'per_label':D1.set_index('label')[['precision','recall','f1']].round(3).to_dict('index')}
    if args.headlines2:
        h2=pd.read_csv(args.headlines2); dd=coded.merge(h2[['sample_id']+['human_'+l for l in LABELS]],on='sample_id',suffixes=('','_c2')).dropna(subset=['human_'+l+'_c2' for l in LABELS])
        ka=[dict(label=l,alpha=kripp(dd['human_'+l].astype(int),dd['human_'+l+'_c2'].astype(int)),pct_agree=float((dd['human_'+l]==dd['human_'+l+'_c2']).mean())) for l in LABELS]
        KA=pd.DataFrame(ka); allv=np.concatenate([dd['human_'+l].values for l in LABELS]); allv2=np.concatenate([dd['human_'+l+'_c2'].values for l in LABELS])
        KA.loc[len(KA)]=dict(label='ALL (pooled)',alpha=kripp(allv.astype(int),allv2.astype(int)),pct_agree=float((allv==allv2).mean()))
        KA.to_csv(os.path.join(outdir,'krippendorff_alpha_headlines.csv'),index=False); summary['headlines']['krippendorff_alpha_pooled']=float(KA.alpha.iloc[-1]); summary['headlines']['n_double_coded']=int(len(dd))

def parse_countries(s):
    if pd.isna(s) or not str(s).strip(): return None
    return [c.strip() for c in re.split(r'[;,|]',str(s)) if c.strip()]
def entities(row,prefix,n=3):
    return [row[f'{prefix}_{i}'] for i in range(1,n+1) if pd.notna(row.get(f'{prefix}_{i}')) and str(row.get(f'{prefix}_{i}')).strip().lower() not in ('','nan','none','null','n/a')]

def score_articles(args,outdir,summary):
    key=pd.read_csv(os.path.join(KEYDIR,'KEY_validation_articles.csv'))
    h=pd.read_csv(args.articles)
    d=key.merge(h.drop(columns=[c for c in ['title','media_name','publish_date','text'] if c in h.columns]),on='sample_id')
    coded=d[(d[[f'human_actor_{i}' for i in range(1,4)]].notna().any(axis=1))|(d[[f'human_source_{i}' for i in range(1,4)]].notna().any(axis=1))]
    out=[]; errs=[]
    for kind,mp,hp in [('actor','top_actor','human_actor'),('source','top_source','human_source')]:
        ent_tp=ent_fp=ent_fn=0
        cc=collections.defaultdict(lambda: dict(tp=0,fp=0,fn=0))
        for _,r in coded.iterrows():
            model=entities(r,mp); human=entities(r,hp)
            # entity-level
            matched=set()
            for m in model:
                hit=[i for i,hh in enumerate(human) if i not in matched and ent_match(m,hh)]
                if hit: matched.add(hit[0]); ent_tp+=1
                else: ent_fp+=1; errs.append(dict(kind=kind,sample_id=r.sample_id,outlet=r.media_name,error_type='model extra (FP)',model_entity=m,model_country=classify(m,kind),human_entities='; '.join(human)))
            for i,hh in enumerate(human):
                if i not in matched: ent_fn+=1; errs.append(dict(kind=kind,sample_id=r.sample_id,outlet=r.media_name,error_type='model missed (FN)',model_entity='',model_country='',human_entities=hh+' ['+str(classify(hh,kind))+']'))
            # country-level multiset
            mc=[classify(m,kind) for m in model]
            hc=parse_countries(r.get(f'{hp}_countries')) or [classify(hh,kind) for hh in human]
            mcount=collections.Counter(mc); hcount=collections.Counter(hc)
            for c in set(mcount)|set(hcount):
                tp=min(mcount[c],hcount[c]); cc[c]['tp']+=tp; cc[c]['fp']+=mcount[c]-tp; cc[c]['fn']+=hcount[c]-tp
        p=ent_tp/(ent_tp+ent_fp) if ent_tp+ent_fp else np.nan; rc=ent_tp/(ent_tp+ent_fn) if ent_tp+ent_fn else np.nan
        out.append(dict(kind=kind,level='entity (fuzzy match)',country='ALL',tp=ent_tp,fp=ent_fp,fn=ent_fn,precision=p,recall=rc,f1=2*p*rc/(p+rc) if p+rc else np.nan))
        for c,v in sorted(cc.items(),key=lambda kv:-kv[1]['tp']):
            p=v['tp']/(v['tp']+v['fp']) if v['tp']+v['fp'] else np.nan; rc=v['tp']/(v['tp']+v['fn']) if v['tp']+v['fn'] else np.nan
            out.append(dict(kind=kind,level='country',country=c,tp=v['tp'],fp=v['fp'],fn=v['fn'],precision=p,recall=rc,f1=2*p*rc/(p+rc) if p and rc and p+rc else np.nan))
    D3=pd.DataFrame(out); D3.to_csv(os.path.join(outdir,'table_D3_actor_source_country_prf.csv'),index=False)
    open(os.path.join(outdir,'table_D3_actor_source_country_prf.md'),'w').write(D3.to_markdown(index=False,floatfmt='.3f'))
    E=pd.DataFrame(errs)
    if len(E):
        E['impact']=E.model_country.isin(['United States','China','Taiwan']).astype(int)+E.human_entities.str.contains('Taiwan|China|United States').astype(int)
        E.sort_values('impact',ascending=False).head(20).drop(columns='impact').to_csv(os.path.join(outdir,'table_D4_annotation_errors.csv'),index=False)
    summary['articles']={'n_coded':int(len(coded)),'table':D3[D3.country.isin(['ALL','United States','China','Taiwan'])].round(3).to_dict('records')}
    if args.articles2:
        h2=pd.read_csv(args.articles2); dd=coded.merge(h2[['sample_id']+[f'human_{k}_{i}' for k in ('actor','source') for i in range(1,4)]],on='sample_id',suffixes=('','_c2'))
        # intercoder reliability at the country-presence level (per article: is Taiwan/China/US present among sources?), nominal alpha
        rows=[]
        for kind in ('actor','source'):
            for c in ('United States','China','Taiwan'):
                a=[int(c in [classify(e,kind) for e in entities(r,f'human_{kind}')]) for _,r in dd.iterrows()]
                b=[int(c in [classify(e,kind) for e in entities(r,f'human_{kind}') ]) for _,r in dd[[f'human_{kind}_{i}_c2' for i in range(1,4)]].rename(columns=lambda x:x[:-3]).iterrows()]
                rows.append(dict(kind=kind,country=c,alpha=kripp(a,b),pct_agree=float(np.mean(np.array(a)==np.array(b)))))
        pd.DataFrame(rows).to_csv(os.path.join(outdir,'krippendorff_alpha_articles.csv'),index=False); summary['articles']['n_double_coded']=int(len(dd))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--headlines'); ap.add_argument('--headlines2'); ap.add_argument('--articles'); ap.add_argument('--articles2'); ap.add_argument('--outdir',default=os.path.join(HERE,'..','revision_outputs','validation_scores'))
    a=ap.parse_args(); os.makedirs(a.outdir,exist_ok=True); summary={}
    if a.headlines: score_headlines(a,a.outdir,summary)
    if a.articles: score_articles(a,a.outdir,summary)
    json.dump(summary,open(os.path.join(a.outdir,'summary.json'),'w'),indent=2,default=float); print(json.dumps(summary,indent=1,default=float))
