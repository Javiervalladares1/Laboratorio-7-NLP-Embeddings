"""Checks matemáticos útiles: fronteras, muestreo, gradientes, equivalencia y splits."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from src.sgns import SGNS,generate_pairs,keep_prob,pair_count
from src.evaluation import keyed,analogia
from src.classification import NewsClassifier,collate
from src.preprocessing import tokenize

ids=np.array([0,1,2,3],dtype=np.int32);off=np.array([0,2,4],dtype=np.int64)
pairs=generate_pairs(ids,off,2)
assert set(map(tuple,pairs))=={(0,1),(1,0),(2,3),(3,2)}
assert len(pairs)==pair_count(off,2)
assert tokenize("Don't re-enter <unk> 12.5!")==['don','t','re','enter','12.5']
p=keep_prob(np.array([10000,100,1]),1e-5)
assert 0<p[0]<p[1]<=p[2]<=1
m=SGNS(4,3);loss=m(torch.tensor([0]),torch.tensor([1]),torch.tensor([[2,3]])).sum()
assert np.isclose(loss.item(),3*np.log(2))
loss.backward();assert m.output.weight.grad is not None
v=keyed(['man','woman','king','queen','other'],np.array([[1,0,0],[1,1,0],[1,0,1],[1,1,1],[-1,0,0]],dtype=np.float32))
own=analogia(v,'man','woman','king');ref=v.most_similar(positive=['woman','king'],negative=['man'],topn=2)
assert [w for w,s in own]==[w for w,s in ref]
assert np.allclose([s for w,s in own],[s for w,s in ref],atol=1e-6)
# Verificar también la fórmula propia de 3CosMul con la implementación de referencia.
from src.evaluation import unit
mat=unit(v.vectors)
a,b,c=[mat[v.key_to_index[w]] for w in ['man','woman','king']]
scores=((mat@b+1)/2)*((mat@c+1)/2)/((mat@a+1)/2+1e-6)
for w in ['man','woman','king']:scores[v.key_to_index[w]]=-np.inf
order=np.argsort(-scores)[:2]
ref_mul=v.most_similar_cosmul(positive=['woman','king'],negative=['man'],topn=2)
assert [v.index_to_key[i] for i in order]==[w for w,s in ref_mul]
assert np.allclose(scores[order],[s for w,s in ref_mul],atol=1e-5)
weights=np.arange(18,dtype=np.float32).reshape(6,3);weights[0]=0
clf=NewsClassifier(weights,True)
flat,offsets,y=collate([(np.array([1,2]),0),(np.array([3]),1)])
assert torch.allclose(clf.emb(flat,offsets),torch.tensor(np.stack([weights[[1,2]].mean(0),weights[3]])))
assert not clf.emb.weight.requires_grad
print('PASS: tokenización, pares sin cruzar líneas, subsampling, pérdida SGNS, gradientes, 3CosAdd/3CosMul y EmbeddingBag/offsets.')
