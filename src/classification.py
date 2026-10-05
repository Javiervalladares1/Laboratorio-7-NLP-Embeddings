"""AG News: selección estratificada en validación y test final sellado."""
import json,time,copy
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,log_loss
from gensim.models import KeyedVectors
from datasets import load_from_disk
from .preprocessing import ROOT,tokenize,dump_json
SEED=23045
LABELS=['World','Sports','Business','Sci/Tech']

class NewsClassifier(nn.Module):
    def __init__(self,weights,freeze=False):
        super().__init__()
        self.emb=nn.EmbeddingBag.from_pretrained(torch.as_tensor(weights),freeze=freeze,mode='mean',sparse=not freeze,padding_idx=0)
        self.mlp=nn.Sequential(nn.Linear(weights.shape[1],128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,4))
    def forward(self,ids,offsets):return self.mlp(self.emb(ids,offsets))

def collate(rows):
    lengths=[len(ids) for ids,label in rows]
    offsets=torch.tensor(np.r_[0,np.cumsum(lengths[:-1])],dtype=torch.long)
    ids=torch.tensor(np.concatenate([ids for ids,label in rows]),dtype=torch.long)
    return ids,offsets,torch.tensor([label for ids,label in rows],dtype=torch.long)

def metrics(y,p):
    precision,recall,f1,_=precision_recall_fscore_support(y,p,average='macro',zero_division=0)
    return {'accuracy':float(accuracy_score(y,p)),'precision_macro':float(precision),'recall_macro':float(recall),'f1_macro':float(f1)}

def encode(texts,mapping):
    return [np.asarray([mapping.get(w,1) for w in t] or [1],dtype=np.int64) for t in texts]

def infer(model,rows,batch=1024):
    model.eval();labels=[];predictions=[];total_loss=0
    with torch.no_grad():
        for ids,offsets,y in DataLoader(rows,batch_size=batch,shuffle=False,collate_fn=collate):
            logits=model(ids,offsets);total_loss+=nn.functional.cross_entropy(logits,y,reduction='sum').item()
            labels.extend(y.tolist());predictions.extend(logits.argmax(-1).tolist())
    return total_loss/len(rows),metrics(labels,predictions),confusion_matrix(labels,predictions,labels=range(4)).tolist()

def train_network(weights,freeze,train_rows,val_rows,name,epochs=10):
    path=ROOT/'results'/f'clf_{name}.json';checkpoint=ROOT/'models'/f'clf_{name}.pt'
    if path.exists() and checkpoint.exists():return json.loads(path.read_text())
    torch.manual_seed(SEED);torch.set_num_threads(4)
    model=NewsClassifier(weights.copy(),freeze)
    head=torch.optim.Adam(model.mlp.parameters(),lr=.003)
    embed=None if freeze else torch.optim.SparseAdam(model.emb.parameters(),lr=.003)
    generator=torch.Generator().manual_seed(SEED)
    train_loader=DataLoader(train_rows,batch_size=512,shuffle=True,generator=generator,collate_fn=collate)
    best=-1;rec=[];start=time.perf_counter()
    for epoch in range(epochs):
        t=time.perf_counter();model.train();loss_sum=0
        for ids,offsets,y in train_loader:
            head.zero_grad(set_to_none=True)
            if embed:embed.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(ids,offsets),y);loss.backward();head.step()
            if embed:embed.step()
            loss_sum+=loss.item()*len(y)
        val_loss,val_metrics,_=infer(model,val_rows)
        row={'epoch':epoch+1,'train_loss':loss_sum/len(train_rows),'val_loss':val_loss,**val_metrics,'seconds':time.perf_counter()-t};rec.append(row)
        if val_metrics['f1_macro']>best:
            best=val_metrics['f1_macro'];best_epoch=epoch+1;best_metrics=val_metrics
            torch.save(model.state_dict(),checkpoint)
    out={'name':name,'freeze':freeze,'epochs':epochs,'best_epoch':best_epoch,'validation':best_metrics,'records':rec,'training_seconds':time.perf_counter()-start,'total_parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'n_train':len(train_rows),'n_val':len(val_rows),'embedding_dim':weights.shape[1],'vocab_size':len(weights),'hardware':'Apple M1 Pro; CPU 4 threads','learning_rate':.003,'batch_size':512}
    dump_json(path,out);print(name,'F1 val',round(best,4),'sec',round(out['training_seconds'],1),flush=True)
    return out

def run():
    torch.set_num_threads(4)
    ds=load_from_disk(str(ROOT/'data/ag_news'))
    texts=[tokenize(t) for t in ds['train']['text']];y=np.array(ds['train']['label'])
    train_idx,val_idx=train_test_split(np.arange(len(y)),test_size=.1,random_state=SEED,stratify=y)
    # Vocabulario downstream sólo construido con entrenamiento, nunca validación/test.
    counts=Counter(w for i in train_idx for w in texts[i]);task_words=sorted(w for w,n in counts.items() if n>=2)
    selection=json.loads((ROOT/'results/selection.json').read_text())
    embeddings={'SGNS':KeyedVectors.load(str(ROOT/'models'/f"{selection['best']}.kv")),'gensim':KeyedVectors.load(str(ROOT/'models/gensim.kv')),'GloVe':KeyedVectors.load(str(ROOT/'data/glove.kv'),mmap='r')}
    mapping={'<pad>':0,'<unk>':1,**{w:i+2 for i,w in enumerate(task_words)}}
    rng=np.random.default_rng(SEED)
    dims=embeddings['SGNS'].vector_size
    # Igual vocabulario de tarea/arquitectura para todos. Los tokens sin vector fuente
    # se mapean a UNK. Se reporta OOV fuente por separado del filtrado min_freq=2.
    weights={'random':rng.normal(0,.1,(len(mapping),dims)).astype(np.float32)};weights['random'][0]=0
    mappings={'random':mapping};encoded={'random':encode(texts,mapping)}
    oov={}
    for name,kv in embeddings.items():
        m={'<pad>':0,'<unk>':1,**{w:mapping[w] for w in task_words if w in kv}}
        mat=np.zeros((len(mapping),kv.vector_size),dtype=np.float32)
        # UNK = media de vectores conocidos en entrenamiento, congelada o entrenable.
        covered=[w for w in task_words if w in kv]
        mat[1]=np.mean(np.stack([kv[w] for w in covered]),axis=0)
        for w in covered:mat[mapping[w]]=kv[w]
        weights[name]=mat;mappings[name]=m;encoded[name]=encode(texts,m)
        oov[name]={}
        for split,ix in [('train',train_idx),('validation',val_idx)]:
            total=sum(len(texts[i]) for i in ix)
            absent=sum(w not in kv for i in ix for w in texts[i]);unk=sum(w not in m for i in ix for w in texts[i])
            oov[name][split]={'tokens':total,'source_oov_percent':100*absent/total,'classifier_unk_percent':100*unk/total}
    dump_json(ROOT/'results/classification_protocol.json',{'seed':SEED,'n_train':len(train_idx),'n_validation':len(val_idx),'n_official_test':len(ds['test']),'split_stratified':True,'task_vocab_size':len(mapping),'task_min_freq':2,'hidden_dim':128,'dropout':.2,'epochs':10,'fracs':[.01,.1,.5,1.0],'vocab_policy':'Vocabulario común definido sólo con train; OOV del embedding fuente -> UNK; filas sin vector inaccesibles. Aleatorio tiene dimensión del SGNS seleccionado.','test_policy':'Una inferencia en test por inicialización y fracción tras seleccionar congelado/fine-tuning y epoch sólo por validación; ninguna selección posterior con test.'})
    selections={};baseline_models={};learning=[]
    for fraction in [.01,.1,.5,1.0]:
        tag=str(int(fraction*100))
        subset=train_idx if fraction==1 else train_test_split(train_idx,train_size=fraction,random_state=SEED,stratify=y[train_idx])[0]
        selections[tag]={}
        for name in ['random','SGNS','gensim','GloVe']:
            train_rows=[(encoded[name][i],int(y[i])) for i in subset];val_rows=[(encoded[name][i],int(y[i])) for i in val_idx]
            variants=[False] if name=='random' else [True,False]
            runs=[]
            for frozen in variants:
                suffix='frozen' if frozen else 'finetune';run_name=f'{name}_{suffix}_p{tag}'
                runs.append(train_network(weights[name],frozen,train_rows,val_rows,run_name))
            chosen=max(runs,key=lambda r:r['validation']['f1_macro'])
            selections[tag][name]=chosen['name']
        # Baseline exacto TF-IDF + regresión logística (sin usar test en vocabulario).
        import joblib
        baseline_path=ROOT/'results'/f'clf_TFIDF_p{tag}.json';model_path=ROOT/'models'/f'clf_TFIDF_p{tag}.joblib'
        if baseline_path.exists() and model_path.exists():
            vec,logreg=joblib.load(model_path)
        else:
            start=time.perf_counter()
            vec=TfidfVectorizer(analyzer=tokenize,ngram_range=(1,1),min_df=2,max_features=60000,sublinear_tf=True)
            xtr=vec.fit_transform([' '.join(texts[i]) for i in subset]);xval=vec.transform([' '.join(texts[i]) for i in val_idx])
            # warm_start permite observar pérdidas de la misma solución de LR sin
            # cambiar de modelo a SGDClassifier. Bloques de 10 iteraciones L-BFGS.
            logreg=LogisticRegression(C=4,max_iter=10,warm_start=True,solver='lbfgs',random_state=SEED)
            rec=[]
            import warnings
            from sklearn.exceptions import ConvergenceWarning
            for block in range(1,7):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',ConvergenceWarning);logreg.fit(xtr,y[subset])
                rec.append({'epoch':block,'train_loss':float(log_loss(y[subset],logreg.predict_proba(xtr))),'val_loss':float(log_loss(y[val_idx],logreg.predict_proba(xval)))})
            validation=metrics(y[val_idx],logreg.predict(xval))
            joblib.dump((vec,logreg),model_path)
            dump_json(baseline_path,{'name':f'TFIDF_p{tag}','validation':validation,'records':rec,'training_seconds':time.perf_counter()-start,'total_parameters':int(logreg.coef_.size+logreg.intercept_.size),'trainable_parameters':int(logreg.coef_.size+logreg.intercept_.size),'n_train':len(subset),'vocab_size':len(vec.vocabulary_),'loss_note':'Entropía cruzada sin término regularizador; bloques de 10 iteraciones L-BFGS, no epochs de SGD.'})
        baseline_models[tag]=(vec,logreg)
        selections[tag]['TFIDF']=f'TFIDF_p{tag}'
        dump_json(ROOT/'results/classifier_selection.json',selections)
    # Test oficial se abre después de terminar toda selección de variantes/epochs.
    test_texts=[tokenize(t) for t in ds['test']['text']];test_y=np.array(ds['test']['label'])
    final_path=ROOT/'results/classification_final.json'
    if final_path.exists():
        print('Test ya evaluado; se reutiliza resultado sellado.',flush=True);return
    for name,kv in embeddings.items():
        total=sum(len(t) for t in test_texts)
        oov[name]['test']={'tokens':total,'source_oov_percent':100*sum(w not in kv for t in test_texts for w in t)/total,'classifier_unk_percent':100*sum(w not in mappings[name] for t in test_texts for w in t)/total}
    final=[]
    for tag,models in selections.items():
        for name,run_name in models.items():
            if name=='TFIDF':
                vec,lr=baseline_models[tag];pred=lr.predict(vec.transform([' '.join(t) for t in test_texts]));test_metrics=metrics(test_y,pred);cm=confusion_matrix(test_y,pred,labels=range(4)).tolist()
            else:
                record=json.loads((ROOT/'results'/f'clf_{run_name}.json').read_text())
                model=NewsClassifier(weights[name],record['freeze']);model.load_state_dict(torch.load(ROOT/'models'/f'clf_{run_name}.pt',weights_only=True))
                rows=list(zip(encode(test_texts,mappings[name]),map(int,test_y)));_,test_metrics,cm=infer(model,rows)
            final.append({'initialization':name,'fraction':int(tag)/100,'selected_run':run_name,'test':test_metrics,'confusion_matrix':cm})
            print('Test',tag,name,round(test_metrics['f1_macro'],4),flush=True)
    dump_json(ROOT/'results/classification_oov.json',oov)
    dump_json(final_path,final)
