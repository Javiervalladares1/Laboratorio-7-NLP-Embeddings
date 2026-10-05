import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.preprocessing import ROOT,dump_json
from src.sgns import CONFIGS,train
from gensim.models import KeyedVectors
words=json.loads((ROOT/'data/vocab.json').read_text());glove=KeyedVectors.load(str(ROOT/'data/glove.kv'),mmap='r')
candidates=[w for w in words if w in glove][:30000]
dump_json(ROOT/'results/shared_vocab.json',candidates)
del glove
all_results=[]
for c in CONFIGS:all_results.append(train(c,candidates))
# Se selecciona también la mejor dimensión 100 para clasificación comparable.
eligible=[r for r in all_results if r['corpus_tokens_known']>=20_000_000]
best=max(eligible,key=lambda r:r['selection_score'])
best100=max((r for r in eligible if r['config']['dim']==100),key=lambda r:r['selection_score'])
dump_json(ROOT/'results/selection.json',{'best':best['config']['name'],'best_100':best100['config']['name'],'criterion':'Modelo principal: >=20M tokens conocidos. Spearman WordSim-353 + accuracy total de analogías compartidas; mejor epoch.', 'scores':{r['config']['name']:r['selection_score'] for r in all_results}})
print('Selección',best['config']['name'],best100['config']['name'],flush=True)
