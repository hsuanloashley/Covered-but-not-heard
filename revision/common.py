import pandas as pd, numpy as np, os, json, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from country_map import classify
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','revision_outputs'); os.makedirs(OUT,exist_ok=True)
WINDOWS=['COVID','Pre-Pelosi','Pelosi','Blinken','Election','Post-election']
WIN_DESC={'COVID':'Jan20-Jun21','Pre-Pelosi':'Jul21-Jul22','Pelosi':'Aug-Dec22','Blinken':'Jan-Aug23','Election':'Sep23-Jun24','Post-election':'Jul24-Mar25'}
TOPICS=['crisis','independence','unification','democratic_values','natural_disaster','semiconductors','cultural']
STANCES=['pro_Taiwan','against_Taiwan','pro_China','against_China']
LABELS=TOPICS+STANCES
ACOLS=['top_actor_1','top_actor_2','top_actor_3']; SCOLS=['top_source_1','top_source_2','top_source_3']
BIAS={'Left':1,'Lean Left':2,'Center':3,'Lean Right':4,'Right':5}
def load():
    M=pd.read_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)),'master.pkl'))
    M['bias_num']=M.allsides.map(BIAS)
    return M
def save_table(df,name,index=True,floatfmt='.3f'):
    df.to_csv(os.path.join(OUT,name+'.csv'),index=index)
    with open(os.path.join(OUT,name+'.md'),'w') as f: f.write(df.to_markdown(index=index,floatfmt=floatfmt))
def save_json(d,name):
    json.dump(d,open(os.path.join(OUT,name+'.json'),'w'),indent=2,default=float)
def long_mentions(A,kind,extended=True):
    cols=ACOLS if kind=='actor' else SCOLS
    L=A[['id','window','publish_date']+cols].melt(id_vars=['id','window','publish_date'],value_name='name').dropna(subset=['name'])
    L=L[L.name.astype(str).str.strip().str.lower().isin(['','none','null','n/a','nan','unknown'])==False]
    L['country']=L.name.apply(lambda x: classify(x,kind,extended))
    return L
def country_shares(L,by=None):
    """share of US/China/Taiwan among mentions assigned to one of the three"""
    L=L[L.country.isin(['United States','China','Taiwan'])]
    if by is None:
        vc=L.country.value_counts(); return (100*vc/vc.sum()).reindex(['United States','China','Taiwan']).round(1), len(L)
    t=pd.crosstab(L[by],L.country,normalize='index')*100
    n=L.groupby(by).size()
    t=t.reindex(columns=['United States','China','Taiwan']).round(1); t['n_mentions']=n
    return t
