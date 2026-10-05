"""Informe de cinco páginas. Todos los valores se leen de resultados ejecutados."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.utils import ImageReader
from src.preprocessing import ROOT
import numpy as np
OUT=ROOT/'output/pdf/Laboratorio_7_Informe.pdf'
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Text',fontName='Helvetica',fontSize=8.4,leading=11,spaceAfter=5,textColor=colors.HexColor('#172b3a')))
styles.add(ParagraphStyle(name='SmallText',parent=styles['Text'],fontSize=7.1,leading=9))
styles.add(ParagraphStyle(name='TitleLab',fontName='Helvetica-Bold',fontSize=19,leading=23,spaceAfter=8,textColor=colors.HexColor('#155e75')))
styles.add(ParagraphStyle(name='HeadingLab',fontName='Helvetica-Bold',fontSize=11,leading=14,spaceBefore=7,spaceAfter=6,textColor=colors.HexColor('#155e75')))
story=[]
def load(name):return json.loads((ROOT/'results'/name).read_text())
def p(text,small=False):story.append(Paragraph(text,styles['SmallText' if small else 'Text']))
def h(text):story.append(Paragraph(text,styles['HeadingLab']))
def image(name,width=495,height=None):
 path=ROOT/'figures'/name;w,h=ImageReader(str(path)).getSize();story.append(Image(str(path),width=width,height=height or width*h/w));story.append(Spacer(1,4))
def table(rows,widths=None,fontsize=7):
 def cell(x):return Paragraph(str(x),ParagraphStyle('cell',fontName='Helvetica',fontSize=fontsize,leading=fontsize+2))
 t=Table([[cell(v) for v in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2eff3')),('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#155e75')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#155e75')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d5dde0')),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]));story.append(t);story.append(Spacer(1,5))
def pct(x):return 'N/D' if x is None else f'{100*x:.1f}'
def f(x):return 'N/D' if x is None else f'{x:.3f}'
def footer(canvas,doc):
 canvas.setStrokeColor(colors.HexColor('#c4d9e1'));canvas.line(36,32,576,32)
 canvas.setFont('Helvetica',7);canvas.setFillColor(colors.HexColor('#48606c'));canvas.drawString(36,20,'CC3092 | NLP y Embeddings | Valladares 23045 / Cumes 23236');canvas.drawRightString(576,20,f'{doc.page} / 5')

def build():
 prep=load('preprocessing.json');sel=load('selection.json');emb=load('embedding_final.json');clf=load('classification_final.json');oov=load('classification_oov.json')
 runs=[load(x.name) for x in sorted((ROOT/'results').glob('sgns_*.json'))]
 selected=next(r for r in runs if r['config']['name']==sel['best']);best=selected['records'][selected['best_epoch']-1];ref=load('gensim_training.json')
 full={r['initialization']:r for r in clf if r['fraction']==1};small={r['initialization']:r for r in clf if r['fraction']==.01}
 repo=load('delivery.json')['repository'] if (ROOT/'results/delivery.json').exists() else 'Repositorio local Git; publicación pendiente'
 # Página 1
 story.append(Paragraph('Laboratorio 7: NLP y embeddings',styles['TitleLab']))
 p('<b>Javier Valladares, 23045 · Ian Cumes, 23236</b><br/>CC3092 - Deep Learning y Sistemas Inteligentes · 05 de octubre de 2026')
 h('1. Pipeline y exploración')
 train=prep['splits']['train']
 p(f"WikiText-103 train: <b>{train['articles']:,} artículos, {train['lines']:,} líneas y {train['tokens']:,} tokens</b> después de normalizar. Validación: {prep['splits']['validation']['tokens']:,}; test: {prep['splits']['test']['tokens']:,}. Artículos = encabezados de nivel uno; líneas = filas, incluidas vacías. Usamos un prefijo de {prep['subset_tokens']:,} tokens ({prep['subset_articles_started']:,} artículos iniciados), que deja <b>{prep['known_tokens']:,} tokens conocidos</b>; no se usa validación/test WikiText para entrenar.")
 p('Normalización: NFKC y minúsculas; puntuación, guiones y apóstrofos separados; números literales; etiquetas y marcadores WikiText eliminados. Sin stopwords, stemming ni lematización. Mantener minúsculas y formas léxicas permite reutilizar GloVe; su tokenización no se supone idéntica, por lo que medimos OOV. Comparación con NLTK Treebank en 12 oraciones: nuestra función fragmenta <i>can\'t</i> en <i>can, t</i> y <i>Dr.</i> en <i>dr</i>; Treebank produce <i>ca, n\'t</i> y conserva abreviaturas/puntuación. También difieren en guiones, decimales y correos (detalle en notebook).')
 cov=prep['coverage_full_train']
 p(f"Zipf es aproximadamente válida: pendiente {prep['zipf_slope_rank_100_10000']:.2f} entre rangos 100-10,000; no una ley exacta en las colas. Top 10 / 1,000 / 30,000 cubre <b>{cov['10']:.2f}% / {cov['1000']:.2f}% / {cov['30000']:.2f}%</b> de los tokens de train completo.")
 table([['min_count','Vocabulario (22M)','Tokens OOV (%)']]+[[r['min_count'],f"{r['vocab_size']:,}",f"{r['oov_percent']:.2f}"] for r in prep['thresholds']],widths=[85,160,145])
 image('zipf.png',width=340)
 h('2. Investigación breve')
 p('<b>Embedding(V,d)</b> devuelve filas por ID; <b>padding_idx</b> excluye padding de gradientes; <b>sparse</b> dispersa gradientes (pesos densos). <b>from_pretrained(..., freeze)</b> controla actualización. <b>EmbeddingBag</b> agrega documentos por offsets iniciales con mean/sum/max; max no admite gradiente disperso. CBOW predice centro desde contexto; skip-gram predice contexto desde centro. SGNS usa W de entrada y W\' de salida, conserva W. Word2Vec es predictivo; GloVe ajusta conteos globales. Investigación ampliada y parámetros gensim en notebook y docs/investigacion.md.',small=True)
 story.append(PageBreak())
 # Página 2
 h('3. Entrenamiento propio y referencia')
 cfg=selected['config']
 p('SGNS propio en PyTorch: dos tablas, negativos con distribución f^0.75, pérdida logística estable y SGD por minibatch (suma de pérdidas). CPU M1 Pro, 4 hilos, 16 GB; GPU no utilizada (pico GPU 0). Todas las iteraciones: 3 epochs, min_count=20, sample=10^-5, batch=1,024, LR lineal 0.025 a 0.001, semilla 23045. Se mantiene un vocabulario fijo para las ablaciones; algunas palabras quedan sin entrenar en los prefijos pequeños.')
 rows=[['Iteración','Tokens M','d / w / k','Epoch*','WS353','Acc. (%)','Tiempo min']]
 for r in runs:
  c=r['config'];e=r['records'][r['best_epoch']-1]
  rows.append([c['name'],f"{r['corpus_tokens_known']/1e6:.2f}",f"{c['dim']} / {c['window']} / {c['negative']}",r['best_epoch'],f(e['wordsim']['spearman']),pct(e['analogies']['accuracy']['total']),f"{r['training_seconds']/60:.1f}"])
 table(rows,widths=[92,52,72,43,53,55,80],fontsize=6.8)
 p(f"Selección: Spearman WS353 + accuracy total, con candidatos idénticos; SimLex queda reservado. Modelo principal <b>{sel['best']}</b>, epoch {selected['best_epoch']}; sólo son elegibles modelos con >=20M tokens conocidos. Mejor 100d: {sel['best_100']}. Se guardan vecinos de king, france, computer, good, january y run en cada epoch.",small=True)
 subs=best['subsampling']
 p(f"p(conservar)=min(1,(sqrt(f/t)+1)t/f). Descarte observado de the/of/and: <b>{subs['the']['discarded_percent']:.2f}% / {subs['of']['discarded_percent']:.2f}% / {subs['and']['discarded_percent']:.2f}%</b>. Reduce dominio de palabras funcionales y costo. Epoch seleccionada: {best['pairs_before_subsampling']:,} pares antes, {best['pairs_after_subsampling']:,} después, ventana {cfg['window']}; no cruzan líneas.")
 image('curvas_embeddings.png',width=490)
 p(f"Gensim usa exactamente corpus y vocabulario del modelo principal y dimensión/ventana/negativos/subsampling/LR equivalentes; total {ref['training_seconds']/60:.1f} min. Difieren actualizaciones por pares con 4 workers y minibatches de PyTorch, así como orden/muestras aleatorias; no se esperan pesos iguales. Una pérdida menor no asegura más analogías: el objetivo local no optimiza ese benchmark. Con k distinto las pérdidas ni siquiera tienen la misma escala.")
 image('analogias_vs_tokens.png',width=330)
 story.append(PageBreak())
 # Página 3
 h('4. Aritmética vectorial, benchmark y paralelismo')
 ev=emb['SGNS']['analogies']['add']
 p(f"3CosAdd: q=norm(-norm(a)+norm(b)+norm(c)), ranking por coseno y exclusión de a,b,c; coincide con most_similar en todas las consultas cubiertas. 3CosMul usa cosenos trasladados a [0,1], con epsilon 10^-6. Benchmark: <b>{ev['candidate_count']:,} candidatos compartidos, {ev['evaluated']:,}/{ev['total_questions']:,} preguntas ({100*ev['coverage']:.1f}%)</b>; las preguntas son las mismas en los tres modelos. Resultados por categoría, Add / Mul en %:")
 rows=[['Categoría','n','SGNS','gensim','GloVe']]
 for i,cat in enumerate(ev['categories']):
  vals=[]
  for name in ['SGNS','gensim','GloVe']:
   vals.append(pct(emb[name]['analogies']['add']['categories'][i]['accuracy'])+' / '+pct(emb[name]['analogies']['mul']['categories'][i]['accuracy']))
  rows.append([cat['category'],cat['evaluated'],*vals])
 table(rows,widths=[186,35,84,84,84],fontsize=6.6)
 p('21 analogías propias en 7 tipos (género, capital, gentilicio, comparativo, superlativo, pasado y plural). Notebook/CSV incluyen top-5 con similitudes, rango y coseno de la respuesta y ranking sin exclusión para cada modelo. Ejemplo man:woman :: king:queen:',small=True)
 rows=[['Modelo','Top-1 excluyendo','Rango queen','Cos(queen,q)','Top-1 sin excluir','Paralelismo medio']]
 for name in ['SGNS','gensim','GloVe']:
  q=emb[name]['individual']['queries'][0]
  rows.append([name,q['top5'][0][0],q['expected_rank'],f(q['expected_cosine']),q['without_exclusion'][0][0],f(emb[name]['individual']['parallelism_mean'])])
 table(rows,widths=[60,90,67,76,96,84],fontsize=6.7)
 p('Para suplir la sección 5.3 ausente, medimos cos(v_b-v_a,v_d-v_c) y residuo relativo en vectores crudos. El paralelismo no implica igual magnitud; el ranking tampoco es igualdad algebraica. Sin exclusión, una palabra de consulta puede dominar: la analogía se interpreta como proximidad aproximada, no como igualdad.')
 image('tsne.png',width=340)
 p('t-SNE de 500 palabras normalizadas; grupos definidos con países, números y WordNet (animales, verbos, artefactos). Hay zonas temáticas y solapamientos; la proyección local y la polisemia impiden interpretar distancias 2D como prueba de relaciones exactas.',small=True)
 story.append(PageBreak())
 # Página 4
 h('5. Pipeline end-to-end: AG News')
 p('Split estratificado: 108,000 train / 12,000 validación; 7,600 test oficial. Misma tokenización. EmbeddingBag(mean) + MLP 128/ReLU/dropout 0.2/4 salidas; 10 epochs, batch 512, Adam para MLP y SparseAdam para tabla (LR 0.003). Se elige epoch y variante por F1 macro de validación; después se infiere una vez en test por ganador y fracción. La curva requiere modelos separados por fracción. Gensim se añade como control downstream. Dimensión: la de cada fuente; random usa la dimensión SGNS.')
 rows=[['Modelo / variante','Acc val','P val','R val','F1 val','F1 test','Total / entrenables','Min']]
 for name in ['TFIDF','random','SGNS','gensim','GloVe']:
  if name=='TFIDF': names=['TFIDF_p100']
  elif name=='random':names=['random_finetune_p100']
  else:names=[name+'_frozen_p100',name+'_finetune_p100']
  for rn in names:
   r=load('clf_'+rn+'.json');v=r['validation'];winner=full[name]['selected_run']==rn
   rows.append([rn.replace('_p100','').replace('finetune','FT'),pct(v['accuracy']),pct(v['precision_macro']),pct(v['recall_macro']),pct(v['f1_macro']),pct(full[name]['test']['f1_macro']) if winner else '-',f"{r['total_parameters']/1e6:.2f}M / {r['trainable_parameters']/1e6:.2f}M",f"{r['training_seconds']/60:.1f}"])
 table(rows,widths=[99,44,39,39,40,43,112,35],fontsize=6.4)
 p('TF-IDF + regresión logística: unigramas, min_df=2, sublinear_tf, hasta 60k atributos, C=4; seis bloques de diez pasos L-BFGS con warm_start. Sus curvas muestran entropía cruzada sin regularización, no epochs de SGD. El test se usa sólo con las variantes ganadoras (guion = no evaluada).',small=True)
 image('curvas_clasificacion.png',width=430)
 image('matrices_confusion.png',width=495)
 p('Matrices con filas reales y columnas predichas; W=World, S=Sports, B=Business, T=Sci/Tech. La pérdida del promedio de embeddings elimina orden; noticias con términos compartidos entre Business y Sci/Tech pueden confundirse.',small=True)
 image('f1_vs_datos.png',width=270)
 story.append(PageBreak())
 # Página 5
 h('6. Comparación final y discusión')
 rows=[['Métrica','SGNS','gensim','GloVe']]
 rows.extend([['Tokens / vocabulario / d',f"{selected['corpus_tokens_known']/1e6:.2f}M / {emb['SGNS']['vocab_size']:,} / {emb['SGNS']['dim']}",f"{ref['corpus_tokens']/1e6:.2f}M / {emb['gensim']['vocab_size']:,} / {emb['gensim']['dim']}",'6,000M / 400,000 / 100'],['Entrenamiento / hardware',f"{selected['training_seconds']/60:.1f} min / M1 Pro CPU",f"{ref['training_seconds']/60:.1f} min / M1 Pro CPU",'Tiempo y hardware del 100d: N/D [4]']])
 for method in ['add','mul']:
  rows.append([f'3Cos{method.title()} sem / sint / total (%)',*[' / '.join(pct(emb[n]['analogies'][method]['accuracy'][k]) for k in ['semantic','syntactic','total']) for n in ['SGNS','gensim','GloVe']]])
 rows.append(['Spearman WS353 / SimLex999',*[f(emb[n]['wordsim']['spearman'])+' / '+f(emb[n]['simlex']['spearman']) for n in ['SGNS','gensim','GloVe']]])
 rows.append(['Cobertura WS353 / SimLex (%)',*[pct(emb[n]['wordsim']['coverage'])+' / '+pct(emb[n]['simlex']['coverage']) for n in ['SGNS','gensim','GloVe']]])
 rows.append(['Coseno medio de diferencias',*[f(emb[n]['individual']['parallelism_mean']) for n in ['SGNS','gensim','GloVe']]])
 rows.append(['AG News OOV train / test (%)',*[f"{oov[n]['train']['source_oov_percent']:.2f} / {oov[n]['test']['source_oov_percent']:.2f}" for n in ['SGNS','gensim','GloVe']]])
 rows.append(['F1 macro test (100% / 1%)',*[f(full[n]['test']['f1_macro'])+' / '+f(small[n]['test']['f1_macro']) for n in ['SGNS','gensim','GloVe']]])
 table(rows,widths=[182,97,97,97],fontsize=6.7)
 p('<b>Analogías individuales representativas:</b> top-1 / rango de la respuesta esperada (ranking excluye la consulta). Top-5 y cosenos completos de las 21 consultas en el notebook.',small=True)
 rows=[['Consulta (respuesta esperada)','SGNS top-1 / rango','gensim top-1 / rango','GloVe top-1 / rango']]
 for idx in [1,3,6,9,12,15,18]:
  q=emb['SGNS']['individual']['queries'][idx]
  label=f"{q['a']}:{q['b']} :: {q['c']}:{q['expected']}"
  cells=[]
  for n in ['SGNS','gensim','GloVe']:
   item=emb[n]['individual']['queries'][idx]
   cells.append(f"{item['top5'][0][0]} / {item['expected_rank']}" if 'top5' in item else 'OOV')
  rows.append([label,*cells])
 table(rows,widths=[182,97,97,97],fontsize=6.4)
 cats=emb['SGNS']['analogies']['add']['categories'];covered=[r for r in cats if r['accuracy'] is not None];strong=max(covered,key=lambda r:r['accuracy']);weak=min(covered,key=lambda r:r['accuracy'])
 p(f"<b>Aritmética:</b> SGNS funciona mejor en {strong['category']} ({pct(strong['accuracy'])}%) y peor en {weak['category']} ({pct(weak['accuracy'])}%). Semántica/sintaxis: {pct(emb['SGNS']['analogies']['add']['accuracy']['semantic'])}% / {pct(emb['SGNS']['analogies']['add']['accuracy']['syntactic'])}%. Sus diferencias no son exactamente paralelas (coseno medio {f(emb['SGNS']['individual']['parallelism_mean'])}); queen no tiene por qué ser la vecina más cercana si se permiten palabras de consulta.")
 size=sorted([r for r in runs if r['config']['dim']==100 and r['config']['window']==2 and r['config']['negative']==3],key=lambda r:r['corpus_tokens_known'])
 acc=[r['records'][r['best_epoch']-1]['analogies']['accuracy']['total'] for r in size]
 p(f"<b>Escala y método:</b> con 25/50/100% del corpus y 100d, accuracy total = {' / '.join(pct(x) for x in acc)}%; GloVe = {pct(emb['GloVe']['analogies']['add']['accuracy']['total'])}%. GloVe usa unas {6000/(selected['corpus_tokens_known']/1e6):.0f} veces más tokens. La tendencia apoya un efecto del corpus, pero no permite atribuir causalmente toda la brecha: cambian también dominio, objetivo, optimización y composición del prefijo.")
 dims=[r for r in runs if r['config']['name'] in ['d50_n100','d100_n100','d300_n100']]
 p('<b>Hiperparámetros:</b> '+ '; '.join(f"d={r['config']['dim']}: {pct(r['records'][r['best_epoch']-1]['analogies']['accuracy']['total'])}%" for r in sorted(dims,key=lambda r:r['config']['dim']))+'. Más dimensiones no garantizan mejoras con tres epochs. Ventana pequeña aproxima contexto sintáctico; grande mezcla contexto temático, pero el efecto debe juzgarse por las categorías medidas. Aumentar negativos cambia el objetivo y el costo; su pérdida no es directamente comparable.')
 base=next(r for r in runs if r['config']['name']=='d100_n100');w5=next(r for r in runs if r['config']['name']=='d100_w5');k5=next(r for r in runs if r['config']['name']=='d100_k5')
 p(f"w=2,k=3: {pct(base['records'][base['best_epoch']-1]['analogies']['accuracy']['total'])}%; w=5,k=3: {pct(w5['records'][w5['best_epoch']-1]['analogies']['accuracy']['total'])}%; w=2,k=5: {pct(k5['records'][k5['best_epoch']-1]['analogies']['accuracy']['total'])}%. Gensim logra {pct(emb['gensim']['analogies']['add']['accuracy']['total'])}%; diferencias de actualización y orden explican parte de la discrepancia sin demostrar un error matemático.")
 winner=max(full,key=lambda n:full[n]['test']['f1_macro'])
 p(f"<b>Clasificación:</b> mejor F1 de test al 100%: {winner} ({f(full[winner]['test']['f1_macro'])}); random = {f(full['random']['test']['f1_macro'])}, TF-IDF = {f(full['TFIDF']['test']['f1_macro'])}. Para SGNS se elige {full['SGNS']['selected_run'].replace('_p100','')}; para GloVe, {full['GloVe']['selected_run'].replace('_p100','')}. Al 1%, random/SGNS/GloVe/TF-IDF = {' / '.join(f(small[n]['test']['f1_macro']) for n in ['random','SGNS','GloVe','TFIDF'])}. Con pocas etiquetas, congelar reduce parámetros; fine-tuning adapta al dominio. La decisión se toma por validación, no por test.")
 p('<b>Límites:</b> una semilla, tres epochs, prefijos temáticamente sesgados y dimensiones potencialmente distintas; sin intervalos de confianza. La tabla de tarea se construye con todo train aun al usar pocas etiquetas. Los benchmarks de similitud tienen coberturas distintas; no deben confundirse con evaluación sobre exactamente los mismos pares. GloVe 100d: tiempo/hardware no inequívocos en las fuentes, por eso N/D. Gensim emitió avisos Cython; la auditoría comprobó vectores finitos (ver audit.json).',small=True)
 p('<b>Fuentes:</b> [1] <link href="https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html">PyTorch: Embedding/EmbeddingBag</link>. [2] <link href="https://radimrehurek.com/gensim/models/word2vec.html">gensim Word2Vec/KeyedVectors</link>. [3] <link href="https://arxiv.org/abs/1310.4546">Mikolov et al. (2013)</link>. [4] <link href="https://nlp.stanford.edu/projects/glove/">GloVe</link> y <link href="https://aclanthology.org/D14-1162/">Pennington et al. (2014)</link>. Datos: Salesforce/wikitext y fancyzhx/ag_news en Hugging Face.',small=True)
 p(f'<b>Repositorio:</b> <link href="{repo}">{repo}</link>' if repo.startswith('https://') else '<b>Repositorio:</b> '+repo,small=True)
 doc=SimpleDocTemplate(str(OUT),pagesize=(612,792),leftMargin=36,rightMargin=36,topMargin=34,bottomMargin=42)
 doc.title='Laboratorio 7 - NLP end-to-end y Embeddings'
 doc.author='Javier Valladares 23045; Ian Cumes 23236'
 doc.build(story,onFirstPage=footer,onLaterPages=footer)
 print(OUT,flush=True)
if __name__=='__main__':build()
