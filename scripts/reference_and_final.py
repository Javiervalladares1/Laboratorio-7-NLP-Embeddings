import sys,json,time,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from gensim.models import Word2Vec,KeyedVectors
from gensim.models.word2vec import LineSentence
from src.preprocessing import ROOT,dump_json
from src.evaluation import benchmark,similarity,individual,CONTROL

selection=json.loads((ROOT/'results/selection.json').read_text())
name=selection['best'];r=json.loads((ROOT/'results'/f'sgns_{name}.json').read_text());c=r['config']
words=json.loads((ROOT/'data/vocab.json').read_text());counts=np.load(ROOT/'data/counts.npy')
candidates=json.loads((ROOT/'results/shared_vocab.json').read_text())
corpus=LineSentence(str(ROOT/'data/corpus.txt'),max_sentence_length=10**9)
result_path=ROOT/'results/gensim_training.json'
if not result_path.exists() or not (ROOT/'models/gensim.kv').exists():
 model=Word2Vec(vector_size=c['dim'],window=c['window'],min_count=c['min_count'],sg=1,hs=0,negative=c['negative'],ns_exponent=.75,sample=c['sample'],alpha=c['lr'],min_alpha=c['min_lr'],workers=4,seed=c['seed'],shrink_windows=False,compute_loss=True)
 model.build_vocab_from_freq(dict(zip(words,map(int,counts))),corpus_count=sum(1 for _ in corpus))
 assert set(model.wv.key_to_index)==set(words)
 rec=[];start=time.perf_counter();best=-1
 for epoch in range(c['epochs']):
  t=time.perf_counter();a=c['lr']+(c['min_lr']-c['lr'])*epoch/c['epochs'];b=c['lr']+(c['min_lr']-c['lr'])*(epoch+1)/c['epochs']
  model.train(corpus,total_examples=model.corpus_count,epochs=1,start_alpha=a,end_alpha=b,compute_loss=True)
  seconds=time.perf_counter()-t
  ev=benchmark(model.wv,candidates)['add'];ws=similarity(model.wv,'wordsim353.tsv')
  raw=model.get_latest_training_loss()
  rec.append({'epoch':epoch+1,'train_seconds':seconds,'loss_reported':raw,'loss_note':'Acumulador aproximado de gensim; no comparar su escala con la pérdida media SGNS. Se reinicia por llamada train.', 'analogies':ev,'wordsim':ws,'neighbors':{w:model.wv.most_similar(w,topn=5) for w in CONTROL},'gpu_peak_bytes':0})
  score=ev['accuracy']['total']+ws['spearman']
  if score>best:
   best=score;best_epoch=epoch+1;model.wv.save(str(ROOT/'models/gensim.kv'))
  print('gensim epoch',epoch+1,'analogy',ev['accuracy']['total'],'WS',ws['spearman'],'sec',seconds,flush=True)
 dump_json(result_path,{'config':c,'records':rec,'best_epoch':best_epoch,'selection_score':best,'training_seconds':sum(x['train_seconds'] for x in rec),'elapsed_seconds':time.perf_counter()-start,'hardware':'Apple M1 Pro; CPU 4 workers','vocab_size':len(words),'corpus_tokens':int(counts.sum())})
sgns=KeyedVectors.load(str(ROOT/'models'/f'{name}.kv'));gensim=KeyedVectors.load(str(ROOT/'models/gensim.kv'));glove=KeyedVectors.load(str(ROOT/'data/glove.kv'),mmap='r')
shared=[w for w in words if w in sgns and w in gensim and w in glove][:30000]
assert shared==candidates
out={}
for label,kv in [('SGNS',sgns),('gensim',gensim),('GloVe',glove)]:
 print('Evaluación final',label,flush=True)
 out[label]={'vocab_size':len(kv),'dim':kv.vector_size,'analogies':benchmark(kv,shared,methods=('add','mul')),'wordsim':similarity(kv,'wordsim353.tsv'),'simlex':similarity(kv,'simlex999.txt'),'individual':individual(kv)}
 assert out[label]['individual']['matches_gensim_all']
 dump_json(ROOT/'results/embedding_final.json',out)
sgns.save_word2vec_format(str(ROOT/'models/mejor_sgns.txt'),binary=False)
print('Vectores exportados',len(sgns),sgns.vector_size,flush=True)
