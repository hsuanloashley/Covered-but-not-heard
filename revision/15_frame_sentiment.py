# (was a6_frame_sentiment.py)
from common import *
from outlet_types import outlet_type
import statsmodels.formula.api as smf, statsmodels.api as sm, warnings; warnings.filterwarnings('ignore')
from scipy.stats import pearsonr
M=load(); res={}
PAIRS=[('unification','pro_China'),('against_Taiwan','pro_China'),('democratic_values','pro_Taiwan'),('crisis','against_China'),('crisis','pro_China'),('pro_China','pro_Taiwan'),('semiconductors','pro_Taiwan')]
def run(df,tag):
    df=df.copy(); df['outlet_type']=df.media_name.map(outlet_type)
    for l in LABELS: df[l]=df[l].astype(int)
    rows=[]
    for x,y in PAIRS:
        r,p=pearsonr(df[x],df[y])
        d=df.groupby('yearmonth')[y].transform(lambda v:0<v.sum()<len(v)); dd=df[d]
        if dd.groupby('yearmonth').size().min()<5 or dd[x].sum()<5: dd=df.copy(); dd['yearmonth']=dd.publish_date.dt.to_period('Q').astype(str); dd=dd[dd.groupby('yearmonth')[y].transform(lambda v:0<v.sum()<len(v))]
        try:
            m=smf.glm(f'{y} ~ {x} + C(yearmonth)',dd,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':dd.media_name.astype('category').cat.codes.values})
            b=m.params[x]; se=m.bse[x]; OR=np.exp(b); lo=np.exp(b-1.96*se); hi=np.exp(b+1.96*se); pp=m.pvalues[x]; n=int(m.nobs)
        except Exception as e: OR=lo=hi=pp=np.nan; n=len(dd)
        both=int(((df[x]==1)&(df[y]==1)).sum())
        rows.append(dict(sample=tag,x=x,y=y,n=len(df),n_x=int(df[x].sum()),n_y=int(df[y].sum()),n_both=both,raw_r=r,raw_p=p,OR_fe_cluster=OR,ci_low=lo,ci_high=hi,p_fe_cluster=pp,n_model=n))
    return pd.DataFrame(rows)
L=M[(M.label_error==False)&M.allsides.notna()]   # outlet needed for clustering; all labeled have media_name anyway
Lall=M[(M.label_error==False)]
P=Lall[Lall.in_paper_labels==True]
T=pd.concat([run(P,'paper4062'),run(Lall,'all7616')]); save_table(T,'R2_R7_frame_sentiment',index=False,floatfmt='.3f')
res['table']=T.round(3).to_dict('records')
semi=T[(T.x=='semiconductors')]; res['R2_semiconductors_x_proTaiwan']=semi.round(3).to_dict('records')
save_json(res,'results_a6'); print(T.to_string())
