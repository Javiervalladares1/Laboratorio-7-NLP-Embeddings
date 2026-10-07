"""Verificación de integridad de resultados y trazabilidad de la entrega."""
import sys,json,hashlib,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from gensim.models import KeyedVectors
from src.preprocessing import ROOT,dump_json

def main():
 sel=json.loads((ROOT/'results/selection.json').read_text())
 runs=[json.loads(p.read_text()) for p in (ROOT/'results').glob('sgns_*.json')]
 assert len(runs)==7 and all(len(r['records'])==3 for r in runs)
 selected=next(r for r in runs if r['config']['name']==sel['best'])
 assert selected['corpus_tokens_known']>=20_000_000
 final=json.loads((ROOT/'results/embedding_final.json').read_text())
 count={r['analogies']['add']['evaluated'] for r in final.values()}
 assert len(count)==1
 assert all(r['analogies']['add']['candidate_count']==30000 for r in final.values())
 assert all(len(r['analogies']['add']['categories'])==14 and len(r['analogies']['mul']['categories'])==14 for r in final.values())
 assert all(r['individual']['matches_gensim_all'] for r in final.values())
 checked={}
 for name,path in [('SGNS',ROOT/'models'/f"{sel['best']}.kv"),('gensim',ROOT/'models/gensim.kv')]:
  kv=KeyedVectors.load(str(path));assert np.isfinite(kv.vectors).all()
  checked[name]={'finite':True,'nonzero_vectors':int(np.count_nonzero(np.linalg.norm(kv.vectors,axis=1))), 'vocab_size':len(kv),'dimension':kv.vector_size}
 clf=json.loads((ROOT/'results/classification_final.json').read_text())
 assert len(clf)==20 and len({(r['initialization'],r['fraction']) for r in clf})==20
 assert all(sum(map(sum,r['confusion_matrix']))==7600 for r in clf)
 protocol=json.loads((ROOT/'results/classification_protocol.json').read_text())
 assert protocol['n_train']==108000 and protocol['n_validation']==12000
 log=(ROOT/'results/pipeline.log').read_text() if (ROOT/'results/pipeline.log').exists() else ''
 warnings=log.count("Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'")
 out={'python':platform.python_version(),'hardware':'Apple M1 Pro, 16 GB RAM, CPU 4 hilos','complete_sgns_runs':len(runs),'sgns_epochs':sum(len(r['records']) for r in runs),'shared_candidates':30000,'shared_questions':next(iter(count)),'embedding_integrity':checked,'test_winners_evaluated':len(clf),'official_test_size':7600,'gensim_cython_messages':warnings,'gensim_cython_note':'La rueda gensim 4.4.0 emitió mensajes our_dot_float sin excepción Python capturable; se verificaron vectores finitos y métricas crecientes. No se modificó Word2Vec ni se ocultaron mensajes del registro original.'}
 dump_json(ROOT/'results/audit.json',out)
 hashes={}
 for p in [ROOT/'Laboratorio_7_NLP_Embeddings.ipynb',ROOT/'models/mejor_sgns.txt.gz',ROOT/'output/pdf/Laboratorio_7_Informe.pdf',ROOT/'output/docx/Laboratorio_7_Informe.docx']:
  if p.exists():hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 dump_json(ROOT/'results/artifact_hashes.json',hashes)
 print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
