"""Skip-gram con negative sampling propio. gensim sólo se usa para exportación/evaluación."""
import time,json,platform,os
import numpy as np
import torch
from torch import nn
from numba import njit
from .preprocessing import ROOT,dump_json
from .evaluation import keyed,benchmark,similarity,CONTROL

class SGNS(nn.Module):
    def __init__(self,vocab_size,dim,sparse=True):
        super().__init__()
        self.input=nn.Embedding(vocab_size,dim,sparse=sparse)
        self.output=nn.Embedding(vocab_size,dim,sparse=sparse)
        nn.init.uniform_(self.input.weight,-.5/dim,.5/dim)
        nn.init.zeros_(self.output.weight)
    def forward(self,center,context,negative):
        x=self.input(center);y=self.output(context);n=self.output(negative)
        positive=(x*y).sum(-1)
        negative_scores=(x[:,None,:]*n).sum(-1)
        # Igual que word2vec: descartar negativos que coinciden con el contexto positivo.
        mask=negative!=context[:,None]
        losses=-(nn.functional.logsigmoid(positive)+(nn.functional.logsigmoid(-negative_scores)*mask).sum(-1))
        return losses

def keep_prob(counts,t=1e-5):
    f=counts/counts.sum()
    return np.minimum(1.0,(np.sqrt(f/t)+1)*t/f)

@njit(cache=True)
def generate_pairs(ids,offsets,window):
    n=0
    for i in range(len(offsets)-1):
        length=offsets[i+1]-offsets[i]
        for d in range(1,window+1):n+=2*max(0,length-d)
    pairs=np.empty((n,2),dtype=np.int32);p=0
    for line in range(len(offsets)-1):
        lo=offsets[line];hi=offsets[line+1]
        for pos in range(lo,hi):
            for ctx in range(max(lo,pos-window),min(hi,pos+window+1)):
                if pos!=ctx:
                    pairs[p,0]=ids[pos];pairs[p,1]=ids[ctx];p+=1
    return pairs

def pair_count(offsets,window):
    lengths=np.diff(offsets)
    return int(sum(2*np.maximum(lengths-d,0).sum() for d in range(1,window+1)))

def corpus_fraction(fraction):
    ids=np.load(ROOT/'data/token_ids.npy',mmap_mode='r');offsets=np.load(ROOT/'data/offsets.npy')
    limit=int(len(ids)*fraction)
    offsets=offsets[offsets<limit];offsets=np.r_[offsets,limit]
    return np.array(ids[:limit]),offsets

def subsample(ids,offsets,counts,t,seed):
    rng=np.random.default_rng(seed);keep=rng.random(len(ids))<keep_prob(counts,t)[ids]
    lengths=np.add.reduceat(keep.astype(np.int32),offsets[:-1]); positive=lengths[lengths>0]
    new_offsets=np.r_[0,np.cumsum(positive,dtype=np.int64)]
    drops={w:{'discarded_percent':float(100*(1-keep[ids==i].mean())),'expected_percent':float(100*(1-keep_prob(counts,t)[i]))} for w,i in [(w,json.loads((ROOT/'data/vocab.json').read_text()).index(w)) for w in ['the','of','and']]}
    return ids[keep],new_offsets,drops

CONFIGS=[
 {'name':'d100_n100','dim':100,'fraction':1.0,'window':2,'negative':3},
 {'name':'d50_n100','dim':50,'fraction':1.0,'window':2,'negative':3},
 {'name':'d300_n100','dim':300,'fraction':1.0,'window':2,'negative':3},
 {'name':'d100_n25','dim':100,'fraction':.25,'window':2,'negative':3},
 {'name':'d100_n50','dim':100,'fraction':.5,'window':2,'negative':3},
 {'name':'d100_w5','dim':100,'fraction':1.0,'window':5,'negative':3},
 {'name':'d100_k5','dim':100,'fraction':1.0,'window':2,'negative':5},
]
for c in CONFIGS:c.update(epochs=3,min_count=20,sample=1e-5,lr=.025,min_lr=.001,batch_size=1024,seed=23045)

def train(config,candidates):
    name=config['name'];result_path=ROOT/'results'/f'sgns_{name}.json'
    if result_path.exists() and (ROOT/'models'/f'{name}.kv').exists() and len(json.loads(result_path.read_text())['records'])==config['epochs']:
        print('Ya entrenado',name,flush=True);return json.loads(result_path.read_text())
    torch.manual_seed(config['seed']);torch.set_num_threads(4)
    counts=np.load(ROOT/'data/counts.npy');words=json.loads((ROOT/'data/vocab.json').read_text())
    ids,offsets=corpus_fraction(config['fraction'])
    # Los conteos del subset determinan tanto p(neg) como el subsampling.
    local_counts=np.bincount(ids,minlength=len(words)).astype(np.float64)
    local_counts=np.maximum(local_counts,1e-20)
    cdf=np.cumsum(local_counts**.75);cdf/=cdf[-1]
    model=SGNS(len(words),config['dim'])
    optimizer=torch.optim.SGD(model.parameters(),lr=config['lr'])
    records=[];best_score=-np.inf;start_total=time.perf_counter()
    for epoch in range(config['epochs']):
        start=time.perf_counter()
        kept,off,drops=subsample(ids,offsets,local_counts,config['sample'],config['seed']+epoch)
        pairs=generate_pairs(kept,off,config['window']);rng=np.random.default_rng(config['seed']+epoch)
        rng.shuffle(pairs,axis=0)
        n=len(pairs);loss_sum=0.;bs=config['batch_size']
        for i in range(0,n,bs):
            progress=(epoch+i/n)/config['epochs']
            lr=config['lr']+(config['min_lr']-config['lr'])*progress
            optimizer.param_groups[0]['lr']=lr
            p=pairs[i:i+bs];negative=np.searchsorted(cdf,rng.random((len(p),config['negative']))).astype(np.int64)
            optimizer.zero_grad(set_to_none=True)
            losses=model(torch.from_numpy(p[:,0].astype(np.int64)),torch.from_numpy(p[:,1].astype(np.int64)),torch.from_numpy(negative))
            loss=losses.sum()  # suma para escala de SGD comparable a actualizaciones por par.
            if not torch.isfinite(loss):raise RuntimeError(f'Pérdida no finita {name} epoch {epoch+1}')
            loss.backward();optimizer.step();loss_sum+=loss.item()
        seconds=time.perf_counter()-start
        kv=keyed(words,model.input.weight.detach().numpy().copy())
        ev_start=time.perf_counter();evaluation=benchmark(kv,candidates)['add']
        ws=similarity(kv,'wordsim353.tsv')
        neighbors={w:kv.most_similar(w,topn=5) if w in kv else [] for w in CONTROL}
        row={'epoch':epoch+1,'loss':loss_sum/n,'train_seconds':seconds,'evaluation_seconds':time.perf_counter()-ev_start,'pairs_before_subsampling':pair_count(offsets,config['window']),'pairs_after_subsampling':n,'retained_tokens':len(kept),'subsampling':drops,'analogies':evaluation,'wordsim':ws,'neighbors':neighbors,'gpu_peak_bytes':0,'gpu_peak_note':'Entrenamiento CPU con gradientes dispersos; GPU no utilizada.'}
        records.append(row)
        # Selección exclusivamente sobre WordSim y questions-words, nunca SimLex/test AG News.
        score=(ws['spearman'] or 0)+evaluation['accuracy']['total']
        if score>best_score:
            best_score=score;kv.save(str(ROOT/'models'/f'{name}.kv'))
            torch.save(model.state_dict(),ROOT/'models'/f'{name}.pt');best_epoch=epoch+1
        out={'config':config,'corpus_tokens_known':len(ids),'vocab_size':len(words),'device':'cpu','hardware':platform.processor() or 'Apple M1 Pro','records':records,'best_epoch':best_epoch,'selection_score':best_score,'elapsed_seconds':time.perf_counter()-start_total,'training_seconds':sum(r['train_seconds'] for r in records)}
        dump_json(result_path,out)
        print(name,'epoch',epoch+1,'loss',round(row['loss'],4),'analogy',round(evaluation['accuracy']['total'],4),'WS',round(ws['spearman'],4),'seg',round(seconds,1),flush=True)
        del pairs,kept,kv
    return out
