# (was build_master.py)
import pandas as pd, json, glob, re, numpy as np
import os
HERE=os.path.dirname(os.path.abspath(__file__))
R=os.path.abspath(os.path.join(HERE,'..'))+os.sep   # project root (contains notebooks/, scraped_data/, revision/)
meta=pd.read_pickle(R+'notebooks/df2.pkl')
meta['publish_date']=pd.to_datetime(meta.publish_date,errors='coerce')
print('meta',meta.shape)

# full text
sc=pd.read_csv(R+'scraped_data/scraped_data.csv',usecols=['sid','title','text','publish_date','authors'])
print('scraped',sc.shape,'with text',sc.text.notna().sum(), 'unique sid',sc.sid.nunique())
sc=sc.drop_duplicates('sid')
sc=sc.rename(columns={'title':'scraped_title','publish_date':'scraped_date'})

# actor/source annotations (rebatch = final identification prompt)
rows=[];bad=0
files=sorted(glob.glob(R+'notebooks/Batching_Jsonl_Files/Final/rebatch_output*.jsonl'))+sorted(glob.glob(R+'notebooks/Batching_Jsonl_Files/Final/Output_*.jsonl'))
for p in files:
    for line in open(p):
        r=json.loads(line)
        try: c=r['response']['body']['choices'][0]['message']['content']
        except Exception: bad+=1; continue
        m=re.search(r'\{.*\}',c,re.S)
        if not m: bad+=1; continue
        try: j=json.loads(m.group(0))
        except Exception: bad+=1; continue
        j['id']=r['custom_id']; j['annot_file']=p.split('/')[-1]
        j['annot_skip']= j.get('skip') is True
        rows.append(j)
ann=pd.DataFrame(rows)
ann=ann.drop_duplicates('id')   # rebatch files listed first take priority
print('annotations unique',ann.shape,'skipped',ann.annot_skip.sum(),'bad',bad)
keep=['id','annot_skip','annot_file']+[c for c in ann.columns if c.startswith('top_')]
ann=ann[keep]

# headline labels
E=pd.concat([pd.read_pickle(f) for f in glob.glob(R+'notebooks/raw_labels/*.pkl')])
E['model']=E.llm_results.apply(lambda r: r.response_metadata.get('model_name'))
def ext(c):
    m=re.search(r'\{.*\}',c,re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None
E['j']=E.llm_results.apply(lambda r: ext(r.content))
E['label_error']=E.j.isna()
L=['crisis','independence','unification','democratic_values','natural_disaster','semiconductors','cultural','pro_Taiwan','against_Taiwan','pro_China','against_China']
lab=pd.DataFrame(E.j.apply(lambda j: {k:(j.get(k) if j else None) for k in L}).tolist())
lab['id']=E.id.values; lab['label_model']=E.model.values; lab['label_error']=E.label_error.values
lab['label_attempted']=True
big=pd.read_pickle(R+'notebooks/big_dataframe.pkl')
lab['in_paper_labels']=lab.id.isin(big.id)
for k in L: lab[k]=pd.to_numeric(lab[k],errors='coerce')
lab=lab.drop_duplicates('id')
print('labels attempted',len(lab),'valid',(~lab.label_error).sum(),'paper',lab.in_paper_labels.sum())

# allsides
dfm=pd.read_csv(R+'notebooks/Media_bias.csv')
dfm['clean_url']=dfm['allsides_media_bias_ratings/publication/source_url'].str.replace(r'https?://(www\.)?','',regex=True).str.rstrip('/')
dfm=dfm.rename(columns={'allsides_media_bias_ratings/publication/media_bias_rating':'allsides','allsides_media_bias_ratings/publication/source_name':'allsides_name'})
dfm=dfm.drop_duplicates('clean_url')[['clean_url','allsides','allsides_name']]
meta['clean_url']=meta.media_name.str.replace(r'https?://(www\.)?','',regex=True).str.rstrip('/')

M=meta.merge(sc,left_on='id',right_on='sid',how='left').merge(ann,on='id',how='left').merge(lab,on='id',how='left').merge(dfm,on='clean_url',how='left')
M['annot_attempted']=M.annot_skip.notna()
M['label_attempted']=M.label_attempted.fillna(False)
print('master',M.shape,'text',M.text.notna().sum(),'annot',M.annot_attempted.sum(),'annot nonskip',(M.annot_skip==False).sum(),'allsides',M.allsides.notna().sum(), M.allsides.value_counts().to_dict())
# event windows
def win(d):
    if pd.isna(d): return None
    if d< pd.Timestamp('2021-07-01'): return 'COVID'
    if d< pd.Timestamp('2022-08-01'): return 'Pre-Pelosi'
    if d< pd.Timestamp('2023-01-01'): return 'Pelosi'
    if d< pd.Timestamp('2023-09-01'): return 'Blinken'
    if d< pd.Timestamp('2024-07-01'): return 'Election'
    return 'Post-election'
M['window']=M.publish_date.apply(win)
M['yearmonth']=M.publish_date.dt.to_period('M').astype(str)
print('duplicate ids in master (should be 0):',int(M.id.duplicated().sum()))
M.to_pickle(os.path.join(HERE,'master.pkl'))   # read by common.load()
print(M.window.value_counts())
