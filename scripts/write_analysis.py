import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.preprocessing import ROOT

def load(name):return json.loads((ROOT/'results'/name).read_text())
def acc(run):return run['records'][run['best_epoch']-1]['analogies']['accuracy']
def pc(x):return f'{100*x:.2f}%'

def main():
 emb=load('embedding_final.json');sel=load('selection.json');clf=load('classification_final.json')
 runs={p.stem.replace('sgns_',''):load(p.name) for p in (ROOT/'results').glob('sgns_*.json')}
 full={r['initialization']:r for r in clf if r['fraction']==1};small={r['initialization']:r for r in clf if r['fraction']==.01}
 categories=[c for c in emb['SGNS']['analogies']['add']['categories'] if c['accuracy'] is not None]
 hi=max(categories,key=lambda r:r['accuracy']);lo=min(categories,key=lambda r:r['accuracy'])
 lines=['# Discusión basada en la ejecución','',f"El modelo principal seleccionado fue **{sel['best']}**, epoch {runs[sel['best']]['best_epoch']}; supera el mínimo de 20 millones de tokens conocidos.",'','## ¿Se cumple la aritmética vectorial?','',f"Se cumple parcialmente como recuperación de vecinos. SGNS tiene accuracy 3CosAdd {pc(emb['SGNS']['analogies']['add']['accuracy']['total'])}; su mejor categoría es {hi['category']} ({pc(hi['accuracy'])}) y su peor categoría cubierta es {lo['category']} ({pc(lo['accuracy'])}). No implica que todos los desplazamientos sean exactos."]
 for n,r in emb.items():
  a=r['analogies']['add']['accuracy'];q=r['individual']['queries'][0]
  lines.extend(['',f"**{n}:** semánticas {pc(a['semantic'])}, sintácticas {pc(a['syntactic'])}. En man:woman :: king:queen, queen queda en rango {q['expected_rank']}, con coseno {q['expected_cosine']:.3f}; el top-1 excluyendo es {q['top5'][0][0]}, y sin excluir es {q['without_exclusion'][0][0]}. El coseno medio entre diferencias en las 21 consultas es {r['individual']['parallelism_mean']:.3f}."])
 lines.extend(['','## ¿Es una igualdad?','',"No. 3CosAdd usa vectores normalizados y un ranking; no exige residuo cero sobre vectores crudos. Al permitir palabras de consulta, la salida puede cambiar a una de ellas. Esto revela que parte de la cercanía procede de los términos presentes en la suma. El coseno entre diferencias mide dirección, y el residuo relativo mide discrepancia de dirección y magnitud; ninguna de las consultas acertadas constituye por sí sola una igualdad algebraica.",'','## Tamaño del corpus y distancia a GloVe',''])
 for key in ['d100_n25','d100_n50','d100_n100']:
  r=runs[key];lines.append(f"- {r['corpus_tokens_known']/1e6:.2f}M tokens, dimensión 100, w=2/k=3: {pc(acc(r)['total'])} de analogías.")
 n=runs[sel['best']]['corpus_tokens_known']
 lines.extend(['',f"GloVe alcanza {pc(emb['GloVe']['analogies']['add']['accuracy']['total'])} con 6,000M tokens, unas {6_000_000_000/n:.1f} veces nuestro corpus. El aumento dentro de la ablación apoya un efecto favorable de más datos, pero los prefijos también añaden temas y cobertura. El SGNS principal puede tener otros hiperparámetros que esa ablación. No se puede repartir causalmente la brecha entre cantidad y método sin entrenar ambos métodos sobre exactamente los mismos datos y presupuesto.",'','## Implementación SGNS frente a gensim','',f"La accuracy total es {pc(emb['SGNS']['analogies']['add']['accuracy']['total'])} en PyTorch y {pc(emb['gensim']['analogies']['add']['accuracy']['total'])} en gensim; WordSim rho es {emb['SGNS']['wordsim']['spearman']:.3f} y {emb['gensim']['wordsim']['spearman']:.3f}, respectivamente. Se igualaron corpus, vocabulario y parámetros del objetivo. PyTorch suma gradientes de un minibatch; gensim actualiza por pares con cuatro workers. También difieren orden y muestras aleatorias. La coincidencia de la función de analogías y las pruebas de pérdida/gradientes verifica componentes matemáticos; no demuestra identidad de sus trayectorias de optimización.",'','## Dimensión, ventana y negativos',''])
 for key in ['d50_n100','d100_n100','d300_n100','d100_w5','d100_k5']:
  r=runs[key];a=acc(r);c=r['config']
  lines.append(f"- d={c['dim']}, w={c['window']}, k={c['negative']}: total {pc(a['total'])}; semánticas {pc(a['semantic'])}; sintácticas {pc(a['syntactic'])}; WordSim {r['records'][r['best_epoch']-1]['wordsim']['spearman']:.3f}.")
 a2=acc(runs['d100_n100']);a5=acc(runs['d100_w5'])
 lines.extend(['',f"Al ampliar w=2 a w=5, la mejora semántica observada es {100*(a5['semantic']-a2['semantic']):.2f} puntos porcentuales y la sintáctica {100*(a5['syntactic']-a2['syntactic']):.2f}. Una ventana amplia aporta coocurrencias temáticas y más pares por epoch; el presupuesto computacional no queda constante. Una dimensión mayor no garantiza mejoras con sólo tres epochs. El número de negativos afecta objetivo, escala de pérdida y tiempo; no se ordenan modelos por pérdidas de distintos k.",'','## Clasificación: preentrenados, aleatorios y TF-IDF',''])
 for fraction,rows in [('100%',full),('1%',small)]:
  lines.append(f"### Con {fraction} de etiquetas disponibles")
  for name in ['TFIDF','random','SGNS','gensim','GloVe']:
   lines.append(f"- {name}: F1 macro test {rows[name]['test']['f1_macro']:.4f}; variante elegida en validación: {rows[name]['selected_run']}.")
 for name in ['SGNS','GloVe','gensim']:
  for frac,tag,rows in [('100%','100',full),('1%','1',small)]:
   frozen=load(f'clf_{name}_frozen_p{tag}.json')['validation']['f1_macro'];ft=load(f'clf_{name}_finetune_p{tag}.json')['validation']['f1_macro']
   lines.extend(['',f"Para {name} con {frac}, F1 de validación congelado={frozen:.4f}, fine-tuning={ft:.4f}; convino {'fine-tuning' if ft>frozen else 'congelar'} según ese criterio. El test no se usó para tomar la decisión."])
 winner=max(full,key=lambda name:full[name]['test']['f1_macro']);winner1=max(small,key=lambda name:small[name]['test']['f1_macro'])
 lines.extend(['',f"Ganador observado al 100%: {winner}; al 1%: {winner1}. Los embeddings preentrenados no se presumen superiores: sus resultados se comparan explícitamente con random y TF-IDF. El vocabulario fijo usa texto no etiquetado de todo train en las fracciones pequeñas; TF-IDF se ajusta sólo con su fracción. No hay intervalos de confianza y las diferencias pequeñas no prueban superioridad estadística.",'','## Límites','',"Una semilla, tres epochs de embeddings, prefijos con sesgo temático y cobertura de pares WordSim/SimLex distinta por fuente. Las analogías sí usan preguntas/candidatos idénticos. La normalización ASCII fragmenta algunos nombres y contracciones. EmbeddingBag pierde orden de palabras. t-SNE es exploratorio. Tiempo/hardware original del archivo GloVe 100d: N/D, no se atribuyen datos de otro archivo. Todas las cifras proceden de los JSON ejecutados; no se usan métricas simuladas."])
 (ROOT/'docs/discusion_resultados.md').write_text('\n'.join(lines)+'\n')
 print('Discusión generada',flush=True)
if __name__=='__main__':main()
