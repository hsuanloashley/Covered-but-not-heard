# (was a2b_article_verification_50.py)
from common import *
import html
V=os.path.join(OUT,'validation'); rng=np.random.RandomState(50)
K=pd.read_csv(os.path.join(V,'KEY_validation_articles.csv')); A=pd.read_csv(os.path.join(V,'validation_articles.csv'))
K=K.merge(A[['sample_id','text']],on='sample_id')
# 50 = 6 windows x 2 crisis ~ 4 each + remainder
parts=[]
for (w,c),g in K.groupby(['window','crisis_label']):
    parts.append(g.sample(n=min(4,len(g)),random_state=rng))
S=pd.concat(parts); S=pd.concat([S,K[~K.sample_id.isin(S.sample_id)].sample(n=50-len(S),random_state=rng)]).sort_values('sample_id')
S[['sample_id']].merge(A,on='sample_id').to_csv(os.path.join(V,'validation_articles_50.csv'),index=False)
def sents(t): return [s.strip() for s in re.split(r'(?<=[.!?])\s+',str(t)) if s.strip()]
def toks(name):
    n=re.sub(r'\s*[\(\[].*?[\)\]]','',str(name)); n=re.split(r'\s+-\s+|,\s+',n)[0]
    ws=[w for w in re.findall(r"[A-Za-z][A-Za-z'’\.-]+",n) if w.lower() not in {'the','of','a','an','and','ministry','government','department','s',"'s"}]
    return ws[:4] or [n]
out=['<meta charset="utf-8"><style>body{font-family:Georgia,serif;max-width:900px;margin:auto;line-height:1.45}mark{background:#ffe66d}.a{color:#1a5fb4}.s{color:#26a269}.miss{color:#c01c28}h2{border-top:2px solid #ccc;padding-top:10px}small{color:#666}.q{margin:4px 0 10px 18px;font-size:0.93em}</style>',
 '<h1>Article verification sheet (N = 50)</h1><p>For each article: GPT-4o\'s actors (blue) and sources (green) with the sentences in which they appear. Your job: for each name, mark ✓ if it is really a main actor / a quoted-or-attributed source, ✗ if not, and add any major actor/source GPT missed (skim the first paragraphs; the most-quoted names are usually there). Record answers in <code>validation_articles_50.csv</code> (human_actor_1..3 / human_source_1..3 = the names you confirm or add).</p>']
for i,r in S.reset_index(drop=True).iterrows():
    ss=sents(r.text); out.append(f'<h2>{i+1}. [{r.sample_id}] {html.escape(str(r.title))}</h2><small>{r.media_name} · {r.publish_date} · {r.window} · crisis={r.crisis_label} · {len(ss)} sentences</small>')
    for kind,cols,cls in [('ACTORS',ACOLS,'a'),('SOURCES',SCOLS,'s')]:
        out.append(f'<h3 class="{cls}">{kind}</h3>')
        for c in cols:
            nm=r[c]
            if pd.isna(nm) or str(nm).strip().lower() in ('','none','null','n/a','nan'): continue
            tk=toks(nm); pat=re.compile('|'.join(re.escape(t) for t in tk),re.I)
            hits=[s for s in ss if pat.search(s)]
            out.append(f'<div><b class="{cls}">{html.escape(str(nm))}</b> <small>({len(hits)} matching sentences)</small></div>')
            if not hits: out.append('<div class="q miss">not found by string match: check text manually</div>'); continue
            # prefer sentences with quote marks for sources
            if kind=='SOURCES': hits=sorted(hits,key=lambda s: -bool(re.search(r'said|says|told|according|stated|"|“',s)))
            for s in hits[:3]: out.append('<div class="q">'+pat.sub(lambda m:'<mark>'+html.escape(m.group(0))+'</mark>',html.escape(s))+'</div>')
    lead=' '.join(ss[:3]); out.append('<details><summary>Opening of article</summary><div class="q">'+html.escape(lead)+'</div></details>')
open(os.path.join(V,'article_verification_50.html'),'w').write('\n'.join(out))
print(len(S), S.window.value_counts().to_dict(), S.crisis_label.value_counts().to_dict())
