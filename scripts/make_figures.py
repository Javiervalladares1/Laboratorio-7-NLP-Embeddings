import sys,json,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
from src.preprocessing import ROOT,dump_json
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
F=ROOT/'figures'
def save(name):plt.tight_layout();plt.savefig(F/name,bbox_inches='tight',dpi=160);plt.close()
def load(name):return json.loads((ROOT/'results'/name).read_text())

def generate():
 freq=np.load(ROOT/'data/zipf.npz')['frequencies'];ranks=np.arange(1,len(freq)+1)
 plt.figure(figsize=(6,3));plt.loglog(ranks,freq,color='#155e75');plt.loglog(ranks[:30000],freq[99]*(ranks[:30000]/100)**-1.1,'--',color='#ed8c23',label='Referencia: pendiente -1.10')
 plt.xlabel('Rango');plt.ylabel('Frecuencia');plt.title('WikiText-103: distribución de Zipf');plt.legend();save('zipf.png')
 runs=[load(p.name) for p in sorted((ROOT/'results').glob('sgns_*.json'))]
 fig,axes=plt.subplots(1,2,figsize=(10,3.5))
 for run in runs:
  label=run['config']['name'];epochs=[r['epoch'] for r in run['records']]
  axes[0].plot(epochs,[r['loss'] for r in run['records']],marker='o',label=label)
  axes[1].plot(epochs,[100*r['analogies']['accuracy']['total'] for r in run['records']],marker='o',label=label)
 axes[0].set(xlabel='Epoch',ylabel='Pérdida media por par',title='SGNS: pérdida')
 axes[1].set(xlabel='Epoch',ylabel='Accuracy (%)',title='Analogías sobre vocabulario compartido');axes[1].legend(fontsize=7,ncol=2);save('curvas_embeddings.png')
 final=load('embedding_final.json')
 size=sorted([r for r in runs if r['config']['dim']==100 and r['config']['window']==2 and r['config']['negative']==3],key=lambda r:r['corpus_tokens_known'])
 plt.figure(figsize=(6,3))
 for method in ['semantic','syntactic','total']:
  plt.plot([r['corpus_tokens_known']/1e6 for r in size],[100*r['records'][r['best_epoch']-1]['analogies']['accuracy'][method] for r in size],'o-',label=method)
 plt.axhline(100*final['GloVe']['analogies']['add']['accuracy']['total'],color='black',linestyle='--',label='GloVe 100d; 6,000M tokens')
 plt.xlabel('Tokens WikiText (millones)');plt.ylabel('Accuracy (%)');plt.title('Tamaño del corpus: misma configuración 100d');plt.legend(fontsize=8);save('analogias_vs_tokens.png')
 categories=[r['category'] for r in final['SGNS']['analogies']['add']['categories']]
 fig,ax=plt.subplots(figsize=(10,4));pos=np.arange(len(categories));width=.25
 for j,name in enumerate(final):
  rows=final[name]['analogies']['add']['categories'];ax.bar(pos+(j-1)*width,[100*(r['accuracy'] or 0) for r in rows],width,label=name)
 ax.set_xticks(pos,[c.replace('gram','g').replace('-','\n') for c in categories],fontsize=7);ax.set_ylabel('Accuracy (%)');ax.legend();save('analogias_categorias.png')
 classification=load('classification_final.json')
 plt.figure(figsize=(6,3.5))
 for name in ['random','SGNS','gensim','GloVe','TFIDF']:
  rows=sorted([r for r in classification if r['initialization']==name],key=lambda r:r['fraction'])
  plt.plot([100*r['fraction'] for r in rows],[r['test']['f1_macro'] for r in rows],'o-',label=name)
 plt.xlabel('Fracción de train AG News (%)');plt.ylabel('F1 macro test');plt.title('Curva de aprendizaje (variantes elegidas por validación)');plt.legend(ncol=2);save('f1_vs_datos.png')
 full=[r for r in classification if r['fraction']==1]
 fig,axes=plt.subplots(1,len(full),figsize=(13,3))
 for ax,row in zip(axes,full):
  cm=np.array(row['confusion_matrix']);ax.imshow(cm,cmap='Blues')
  for i in range(4):
   for j in range(4):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=7,color='white' if cm[i,j]>cm.max()/2 else 'black')
  ax.set_title(row['initialization']);ax.set_xticks(range(4),['W','S','B','T']);ax.set_yticks(range(4),['W','S','B','T']);ax.set_xlabel('Predicción')
 axes[0].set_ylabel('Clase real');save('matrices_confusion.png')
 clf_runs=[load(p.name) for p in sorted((ROOT/'results').glob('clf_*.json')) if p.stem.endswith('p100')]
 fig,axes=plt.subplots(1,2,figsize=(10,3.5))
 for r in clf_runs:
  name=r['name'];x=[e['epoch'] for e in r['records']]
  axes[0].plot(x,[e['train_loss'] for e in r['records']],label=name)
  axes[1].plot(x,[e['val_loss'] for e in r['records']],label=name)
 axes[0].set(title='Clasificación: train',xlabel='Epoch / bloque L-BFGS',ylabel='Entropía cruzada')
 axes[1].set(title='Clasificación: validación',xlabel='Epoch / bloque L-BFGS',ylabel='Entropía cruzada');axes[1].legend(fontsize=6,ncol=2);save('curvas_clasificacion.png')
 # Exportar tablas completas, no sólo las filas incluidas en el informe de 5 páginas.
 iterations=[]
 for r in runs:
  cfg=r['config'];e=r['records'][r['best_epoch']-1]
  iterations.append({**cfg,'tokens':r['corpus_tokens_known'],'best_epoch':r['best_epoch'],'loss':e['loss'],'accuracy_semantic':e['analogies']['accuracy']['semantic'],'accuracy_syntactic':e['analogies']['accuracy']['syntactic'],'accuracy_total':e['analogies']['accuracy']['total'],'wordsim':e['wordsim']['spearman'],'training_seconds':r['training_seconds']})
 pd.DataFrame(iterations).to_csv(ROOT/'results/iterations.csv',index=False)
 benchmark_rows=[];individual_rows=[]
 for model,r in final.items():
  for method,ev in r['analogies'].items():
   for cat in ev['categories']:benchmark_rows.append({'model':model,'method':method,**cat})
  for q in r['individual']['queries']:individual_rows.append({'model':model,**q})
 pd.DataFrame(benchmark_rows).to_csv(ROOT/'results/benchmark_categories.csv',index=False)
 pd.DataFrame(individual_rows).to_csv(ROOT/'results/individual_analogies.csv',index=False)
 pd.DataFrame([{k:v for k,v in r.items() if k!='records'} for r in clf_runs]).to_csv(ROOT/'results/classification_validation.csv',index=False)
 pd.DataFrame([{**{k:v for k,v in r.items() if k not in ['test','confusion_matrix']},**r['test']} for r in classification]).to_csv(ROOT/'results/classification_test.csv',index=False)
 # 500 palabras de grupos temáticos con selección reproducible, sin ajustar modelo.
 selection=load('selection.json');kv=KeyedVectors.load(str(ROOT/'models'/f"{selection['best']}.kv"))
 import nltk
 nltk.download('wordnet',quiet=True)
 from nltk.corpus import wordnet as wn
 ranked=json.loads((ROOT/'data/vocab.json').read_text())
 animal={lemma.name() for syn in wn.all_synsets('n') if syn.lexname()=='noun.animal' for lemma in syn.lemmas()}
 artifact={lemma.name() for syn in wn.all_synsets('n') if syn.lexname()=='noun.artifact' for lemma in syn.lemmas()}
 verbs={lemma.name() for syn in wn.all_synsets('v') for lemma in syn.lemmas()}
 countries=set('france germany italy spain portugal england scotland wales ireland japan china india korea vietnam thailand russia ukraine poland sweden norway denmark finland austria belgium switzerland greece turkey egypt morocco algeria tunisia nigeria kenya ethiopia sudan uganda angola zambia zimbabwe botswana namibia madagascar australia canada mexico guatemala belize honduras nicaragua panama cuba jamaica haiti brazil argentina chile peru ecuador bolivia colombia venezuela uruguay paraguay israel iran iraq syria jordan lebanon pakistan afghanistan bangladesh nepal bhutan indonesia malaysia singapore philippines taiwan mongolia kazakhstan uzbekistan turkmenistan kyrgyzstan tajikistan romania bulgaria hungary serbia croatia slovenia slovakia estonia latvia lithuania iceland luxembourg malta cyprus albania armenia georgia azerbaijan qatar bahrain kuwait oman yemen'.split())
 groups={'Países':[w for w in ranked if w in countries],'Números':[w for w in ranked if re.fullmatch(r'[0-9]+',w)],'Animales':[w for w in ranked if w in animal and w not in countries],'Verbos':[w for w in ranked if w in verbs and w not in animal and w not in countries],'Artefactos':[w for w in ranked if w in artifact and w not in verbs and w not in animal]}
 chosen=[];labels=[];used=set()
 for group,cands in groups.items():
  limit=100 if group!='Países' else min(100,len(cands))
  for w in cands:
   if w not in used and w in kv:
    chosen.append(w);labels.append(group);used.add(w)
    if labels.count(group)>=limit:break
 # Si faltan países de una sola palabra, completar con artefactos manteniendo total 500.
 for w in groups['Artefactos']:
  if len(chosen)>=500:break
  if w not in used:chosen.append(w);labels.append('Artefactos');used.add(w)
 assert len(chosen)==500
 x=np.stack([kv[w] for w in chosen]);x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-9)
 coords=TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',random_state=SEED,max_iter=1000).fit_transform(x)
 dump_json(ROOT/'results/tsne_words.json',[{'word':w,'group':g,'x':float(p[0]),'y':float(p[1])} for w,g,p in zip(chosen,labels,coords)])
 fig,ax=plt.subplots(figsize=(8,5))
 for group in groups:
  mask=np.asarray(labels)==group;ax.scatter(coords[mask,0],coords[mask,1],s=18,alpha=.65,label=f'{group} ({mask.sum()})')
 for i in range(0,500,13):ax.annotate(chosen[i],coords[i],fontsize=6,alpha=.8)
 ax.legend(fontsize=8);ax.set_title('t-SNE: 500 palabras del SGNS seleccionado');ax.set_xticks([]);ax.set_yticks([]);save('tsne.png')
 print('Figuras y tablas exportadas',flush=True)
SEED=23045
if __name__=='__main__':generate()
