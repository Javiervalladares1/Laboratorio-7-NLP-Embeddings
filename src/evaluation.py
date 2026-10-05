"""Evaluación propia con normalización, exclusión y denominadores explícitos."""
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from gensim.models import KeyedVectors
from .preprocessing import ROOT,dump_json
CONTROL=['king','france','computer','good','january','run']

def unit(x):
    x=np.asarray(x,dtype=np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def read_analogies():
    out=[];cat=None
    for line in (ROOT/'data/questions-words.txt').read_text().splitlines():
        if line.startswith(':'):cat=line[2:]
        else:
            a,b,c,d=line.lower().split();out.append((cat,a,b,c,d))
    return out

def similarity(kv,filename):
    rows=[];total=0
    for line in (ROOT/'data'/filename).read_text().splitlines():
        if line.startswith('#') or not line.strip(): continue
        fields=line.split()
        try:s=float(fields[2])
        except (ValueError,IndexError):continue
        total+=1;a,b=fields[0].lower(),fields[1].lower()
        if a in kv and b in kv:
            rows.append((s,float(unit(kv[a])@unit(kv[b]))))
    rho=float(spearmanr(np.array(rows)[:,0],np.array(rows)[:,1]).statistic) if len(rows)>2 else None
    return {'spearman':rho,'evaluated':len(rows),'total':total,'coverage':len(rows)/total}

def analogia(kv,a,b,c,k=5,exclude=True,candidates=None):
    """a:b :: c:?; 3CosAdd usa -unit(a)+unit(b)+unit(c), igual a most_similar."""
    keys=kv.index_to_key if candidates is None else candidates
    matrix=unit(kv.vectors) if candidates is None else unit(np.stack([kv[w] for w in keys]))
    q=unit(-unit(kv[a])+unit(kv[b])+unit(kv[c])); scores=matrix@q
    if exclude:
        ids={w:i for i,w in enumerate(keys)}
        for w in {a,b,c}:
            if w in ids:scores[ids[w]]=-np.inf
    order=np.argsort(-scores)[:min(k,int(np.isfinite(scores).sum()))]
    return [(keys[i],float(scores[i])) for i in order]

def benchmark(kv,candidates,methods=('add',),batch=256):
    """Misma lista de candidatos y preguntas; no evaluar consultas parcialmente OOV."""
    keys={w:i for i,w in enumerate(candidates)}
    matrix=unit(np.stack([kv[w] for w in candidates])); questions=read_analogies()
    qs=[q for q in questions if all(w in keys for w in q[1:])]
    indices=np.asarray([[keys[w] for w in q[1:]] for q in qs],dtype=int)
    cats=list(dict.fromkeys(q[0] for q in questions));out={}
    totals={cat:sum(q[0]==cat for q in questions) for cat in cats}
    # BLAS por lotes: memoria O(batch*V), no O(preguntas*V).
    for method in methods:
        correct=np.zeros(len(qs),dtype=bool)
        for start in range(0,len(qs),batch):
            ids=indices[start:start+batch];a,b,c,d=ids.T
            if method=='add':scores=unit(-matrix[a]+matrix[b]+matrix[c])@matrix.T
            else:
                ca=(matrix[a]@matrix.T+1)/2; cb=(matrix[b]@matrix.T+1)/2; cc=(matrix[c]@matrix.T+1)/2
                scores=cb*cc/(ca+1e-6)
            for j in range(3):scores[np.arange(len(ids)),ids[:,j]]=-np.inf
            correct[start:start+len(ids)]=scores.argmax(1)==d
        by_category=[]
        for cat in cats:
            mask=np.asarray([q[0]==cat for q in qs]);n=int(mask.sum());hits=int(correct[mask].sum())
            by_category.append({'category':cat,'total':totals[cat],'evaluated':n,'coverage':n/totals[cat],'correct':hits,'accuracy':hits/n if n else None})
        overall={}
        for name,predicate in [('semantic',lambda q:not q[0].startswith('gram')),('syntactic',lambda q:q[0].startswith('gram')),('total',lambda q:True)]:
            mask=np.array([predicate(q) for q in qs]);n=int(mask.sum());overall[name]=float(correct[mask].mean()) if n else None
        out[method]={'accuracy':overall,'categories':by_category,'evaluated':len(qs),'total_questions':len(questions),'coverage':len(qs)/len(questions),'candidate_count':len(candidates)}
    return out

OWN=[
('genero','man','woman','king','queen'),('genero','boy','girl','brother','sister'),('genero','father','mother','son','daughter'),
('capital','france','paris','germany','berlin'),('capital','italy','rome','japan','tokyo'),('capital','spain','madrid','england','london'),
('gentilicio','france','french','germany','german'),('gentilicio','japan','japanese','china','chinese'),('gentilicio','italy','italian','spain','spanish'),
('comparativo','good','better','bad','worse'),('comparativo','small','smaller','large','larger'),('comparativo','fast','faster','slow','slower'),
('superlativo','small','smallest','large','largest'),('superlativo','good','best','bad','worst'),('superlativo','high','highest','low','lowest'),
('pasado','run','ran','swim','swam'),('pasado','go','went','see','saw'),('pasado','walk','walked','play','played'),
('plural','cat','cats','dog','dogs'),('plural','car','cars','book','books'),('plural','child','children','man','men')]

def individual(kv):
    matrix=unit(kv.vectors);out=[]; agreements=[]
    for typ,a,b,c,d in OWN:
        row={'type':typ,'a':a,'b':b,'c':c,'expected':d}
        if not all(w in kv for w in [a,b,c,d]):
            row['oov']=[w for w in [a,b,c,d] if w not in kv];out.append(row);continue
        q=unit(-unit(kv[a])+unit(kv[b])+unit(kv[c]));scores=matrix@q
        raw_top=np.argsort(-scores)[:5]
        row['without_exclusion']=[(kv.index_to_key[i],float(scores[i])) for i in raw_top]
        row['expected_cosine']=float(scores[kv.key_to_index[d]])
        for w in {a,b,c}:scores[kv.key_to_index[w]]=-np.inf
        ix=np.argsort(-scores)[:5]
        row['top5']=[(kv.index_to_key[i],float(scores[i])) for i in ix]
        row['expected_rank']=int(1+np.count_nonzero(scores>scores[kv.key_to_index[d]]))
        ref=kv.most_similar(positive=[b,c],negative=[a],topn=5)
        agreement=[w for w,s in row['top5']]==[w for w,s in ref] and np.allclose([s for w,s in row['top5']],[s for w,s in ref],atol=2e-6)
        row['matches_gensim']=bool(agreement);agreements.append(agreement)
        # Paralelismo con diferencias de vectores crudos: b-a y d-c.
        x=kv[b]-kv[a];y=kv[d]-kv[c]
        row['difference_cosine']=float(unit(x)@unit(y))
        row['relative_residual']=float(np.linalg.norm(x-y)/max(np.linalg.norm(x)+np.linalg.norm(y),1e-12))
        out.append(row)
    valid=[r for r in out if 'difference_cosine' in r]
    return {'queries':out,'matches_gensim_all':bool(all(agreements)),'parallelism_mean':float(np.mean([r['difference_cosine'] for r in valid])),'top1_accuracy':float(np.mean([r['expected_rank']==1 for r in valid])),'oov_count':len(out)-len(valid)}

def keyed(words,vectors):
    kv=KeyedVectors(vectors.shape[1]);kv.add_vectors(words,np.asarray(vectors,dtype=np.float32));return kv
