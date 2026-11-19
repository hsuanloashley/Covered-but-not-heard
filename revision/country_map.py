"""Country assignment for GPT-4o actor/source strings.
Level 1 = paper's dictionaries (actor_country_mapping / source_country_mapping from Batching.ipynb).
Level 2 = extended: strip role suffix, apply consolidation_dict + dictionaries + keyword rules."""
import re, pandas as pd
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _mapping_cells import consolidation_dict, actor_country_mapping, source_country_mapping
US='United States'; CN='China'; TW='Taiwan'
TW_KW=['taiwan','taipei','tsai ing','lai ching','hsiao','joseph wu','chiu kuo','kuomintang','kmt','democratic progressive','dpp','tsmc','foxconn','ko wen','han kuo','su tseng','audrey tang','chen shih','wellington koo','cheng wen','central epidemic','mainland affairs council','terry gou','eric chu','hou yu','chiang wan','wang ting-yu','executive yuan','legislative yuan','central news agency','united daily','liberty times','formosa','republic of china','mediatek','evergreen','hon hai','chen chien','lin chia','lin fei','wu chao','pan wen','koo li','ma ying','chang mei','taichung','kaohsiung','hsinchu','tseng','taoyuan','lee teng','nationalist party','mark liu','c.c. wei','morris chang','wendell huang','chunghwa','david wenwei chou','david chou','ministry of economic affairs','ministry of foreign affairs','coast guard administration','ocean affairs council','national security bureau','presidential office']
CN_KW=['china','chinese','beijing','xi jinping','pla','people\'s liberation','people’s liberation','wang yi','wang wenbin','zhao lijian','mao ning','hua chunying','qin gang','zhu fenglian','ma xiaoguang','global times','xinhua','cctv','communist party','ccp','eastern theater','wu qian','wei fenghe','li shangfu','dong jun','taiwan affairs office','prc','huawei','hong kong','li keqiang','li qiang','tan kefei','chen binhua','lin jian','guo jiakun','state council','shanghai','ministry of state security','wang huning','xie feng','cui tiankai','zhang jun','geng shuang','chen xiaodong','liu pengyu','people\'s daily','people’s daily','ren guoqiang','zhang youxia','yang jiechi','zhong nanshan']
US_KW=['u.s.','us ','u.s','united states','american','america','biden','trump','white house','pentagon','state department','congress','senate','house ','senator','sen.','rep.','blinken','pelosi','austin','sullivan','kirby','pompeo','mccarthy','mccaul','gallagher','raimondo','psaki','jean-pierre','ned price','kurt campbell','milley','nikki haley','cotton','rubio','graham','cruz','hawley','vance','harris','obama','rick scott','fbi','cia','navy','indo-pacific command','7th fleet','federal reserve','commerce department','defense department','treasury','national security council','nsc','aukus','nvidia','intel','apple','microsoft','google','amazon','lockheed','raytheon','boeing','csis','heritage foundation','brookings','council on foreign relations','rand corp','aei','hudson institute','atlantic council','german marshall','cato','wall street','nasdaq','dow jones','bloomberg','cnn','fox news','nbc','cbs','abc news','new york times','washington post','wall street journal','reuters','associated press',' ap','politico','the hill','axios','npr','cnbc','breitbart','daily caller','newsmax','msnbc','business insider','benzinga','rtt','forbes','fortune','newsweek','time magazine','yahoo','voice of america','radio free asia','american institute in taiwan','ait','musk','tesla','spacex','jake','gordon chang','bonnie glaser','elbridge','fed ','fomc','sec ','ustr','trade representative','hegseth','waltz','wray','gina','jensen huang','pat gelsinger','tim cook','jpmorgan','goldman','morgan stanley','moody','s&p','fitch','usgs','geological survey','cdc','fda','nih','nasa','u.n. ambassador','vance','walz','desantis','ramaswamy','pence','kevin','mike johnson','schumer','mcconnell','jeffries','menendez','risch','wicker','reed','shaheen','ernst','duckworth','markey','sherman','chabot','krishnamoorthi','moulton','kinzinger','cheney','tillis','coons','van hollen','kaine','murphy','lee ','scott ','gop','republican','democrat','blackburn','yellen','gingrich','john cheng','ely ratner','oriana','chris miller','pew research','johns hopkins','60 minutes','institute for supply management']
def strip_role(x):
    x=str(x).lower().strip()
    x=re.sub(r'\s*[\(\[].*?[\)\]]\s*',' ',x)
    x=re.split(r'\s+-\s+|\s+–\s+|,\s+|\s+\|\s+|:\s+',x)[0]
    return x.strip(' .')
def classify(raw, kind='actor', extended=True):
    if pd.isna(raw): return None
    n=str(raw).lower().strip()
    mapping = actor_country_mapping if kind=='actor' else source_country_mapping
    if n in mapping and mapping[n]: return mapping[n]
    if not extended: return None
    n2=strip_role(n)
    n2=consolidation_dict.get(n2,n2)
    for m in (actor_country_mapping,source_country_mapping):
        if n2 in m and m[n2]: return m[n2]
    if n2 in ('china',): return CN
    if n2 in ('us','u.s','usa','united states','washington'): return US
    if n2 in ('taiwan','taipei'): return TW
    s=' '+n2+' '
    # order: Taiwan keywords first (many Taiwan entities contain 'china' e.g. 'republic of china' handled above; 'taiwan affairs office' is PRC -> handled by CN list before TW? ensure)
    if 'taiwan affairs office' in s or 'mainland china' in s: return CN
    if any(k in s for k in TW_KW): return TW
    if any(k in s for k in CN_KW): return CN
    if any(k in s for k in US_KW): return US
    return 'Other/Unmapped'
