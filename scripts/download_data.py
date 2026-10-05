from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,time
from datasets import load_dataset
import gensim.downloader as api
from gensim.test.utils import datapath
ROOT=Path(__file__).resolve().parents[1]
def dataset(name, config, target):
 p=ROOT/'data'/target
 if p.exists(): return
 print('Descargando',name,flush=True)
 d=load_dataset(name,config) if config else load_dataset(name)
 d.save_to_disk(str(p)); print('Guardado',target,{k:len(v) for k,v in d.items()},flush=True)
def glove():
 p=ROOT/'data'/'glove.kv'
 if p.exists(): return
 print('Descargando GloVe',flush=True)
 v=api.load('glove-wiki-gigaword-100'); v.save(str(p))
 print('GloVe guardado',len(v),flush=True)
with ThreadPoolExecutor(3) as pool:
 tasks=[pool.submit(dataset,'Salesforce/wikitext','wikitext-103-raw-v1','wikitext'),pool.submit(dataset,'fancyzhx/ag_news',None,'ag_news'),pool.submit(glove)]
 for t in tasks: t.result()
for name in ['questions-words.txt','wordsim353.tsv','simlex999.txt']:
 import shutil
 shutil.copy(datapath(name),ROOT/'data'/name)
print('Datos listos',flush=True)
