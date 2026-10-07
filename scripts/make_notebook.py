"""Construye un notebook comentado con código, resultados y análisis."""
import sys,json,inspect
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nbformat as nbf
from src.preprocessing import ROOT
from src.sgns import SGNS,keep_prob,generate_pairs
from src.evaluation import unit,analogia
from src.classification import NewsClassifier,collate
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))
def file(name):return (ROOT/name).read_text()
md('''# Laboratorio 7: NLP end-to-end y embeddings
**Javier Valladares (23045) e Ian Cumes (23236)**  
CC3092 · 05/10/2026

Este notebook documenta el pipeline completo y muestra los resultados de entrenamientos reales guardados en `results/`. Los módulos `src/` forman parte de la entrega y contienen el código comentado. La implementación SGNS propia utiliza PyTorch; gensim Word2Vec se usa únicamente como referencia separada.

**Ejecución:** instalar `requirements.txt`, abrir desde la raíz del repositorio y ejecutar todas las celdas. Por defecto se revisa la ejecución guardada. Para reconstruir corpus/modelos, activar `REENTRENAR=True` y ejecutar el bloque de reproducción. Los resultados de test ya existentes se reutilizan para mantener su evaluación única; para un experimento nuevo debe usarse otro directorio de resultados. No se reemplazan resultados faltantes por números inventados.''')
code('''from pathlib import Path
import json, sys, subprocess, platform, inspect, re, unicodedata
import numpy as np
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from numba import njit
from IPython.display import display, Markdown, Image
ROOT = Path.cwd()
assert (ROOT / 'src').is_dir(), 'Abrir desde la raíz del repositorio.'
# Descomprimir los vectores distribuidos en GitHub, si hace falta.
if not (ROOT / 'models/mejor_sgns.txt').exists() and (ROOT / 'models/mejor_sgns.txt.gz').exists():
    import gzip, shutil
    with gzip.open(ROOT / 'models/mejor_sgns.txt.gz', 'rb') as source, (ROOT / 'models/mejor_sgns.txt').open('wb') as target:
        shutil.copyfileobj(source, target)
sys.path.insert(0, str(ROOT))
REENTRENAR = False
def resultado(name):
    return json.loads((ROOT / 'results' / name).read_text())
def figura(name):
    display(Image(filename=str(ROOT / 'figures' / name)))
print('Python', platform.python_version(), '| PyTorch', torch.__version__)
print('MPS disponible:', torch.backends.mps.is_available(), '| SGNS ejecutado en CPU, 4 hilos')
''')
md('''## 1. Datos y protocolo
Se descargan **Salesforce/wikitext, wikitext-103-raw-v1**, **fancyzhx/ag_news** y **glove-wiki-gigaword-100**. Las descargas están en `data/` (omitido de Git). El subconjunto es un prefijo determinista de 22 millones de tokens normalizados de train; después de min_count=20 quedan 21,336,925. Las ventanas respetan líneas. El prefijo puede introducir sesgo temático; las fracciones menores son ablaciones.

Se mantiene el mismo corpus filtrado y vocabulario para el SGNS principal y la referencia gensim. Para las ablaciones de tamaño se conserva el vocabulario del subconjunto completo, haciendo explícito que algunas filas quedan sin entrenar. Los benchmarks comparten candidatos.''')
code('''# Reproducción del pipeline completo. No se ejecuta entrenamiento al revisar resultados.
if REENTRENAR:
    subprocess.run([sys.executable, 'scripts/run_all.py'], check=True)
prep = resultado('preprocessing.json')
display(pd.DataFrame(prep['splits']).T)
print('Subconjunto normalizado:', prep['subset_tokens'])
print('Tokens conocidos:', prep['known_tokens'], '| vocabulario:', prep['vocab_size'])
assert prep['known_tokens'] >= 20_000_000
''')
md('''## 2. Exploración, normalización y tokenización
NFKC/minúsculas y regex ASCII para inglés (los acentos pueden fragmentar nombres); separación de puntuación, apóstrofos y guiones; números literales; eliminación de etiquetas especiales y marcadores WikiText. No stopwords, stemming o lematización. Minúsculas/formas léxicas son compatibles con GloVe uncased, pero sus reglas exactas pueden diferir: se reporta OOV. El conteo de artículos usa encabezados de nivel uno (`= título =`), no cada subsección.

El regex conserva decimales escritos con punto/coma y fragmenta contracciones. Esta es una simplificación explícita: puede producir `t` y `s` de las contracciones. Treebank de NLTK separa sufijos como `n't`, mantiene cierta puntuación y trata abreviaturas de otra manera.''')
code('''TOKEN_RE = re.compile(r"[a-z]+|[0-9]+(?:[.,][0-9]+)*")
def normalize(text):
    text = unicodedata.normalize('NFKC', text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'@[-.,]@', ' ', text)
def tokenize(text):
    return TOKEN_RE.findall(normalize(text))
from nltk.tokenize import TreebankWordTokenizer
comparison = resultado('tokenizer_comparison.json')
assert len(comparison) >= 10
display(pd.DataFrame(comparison))
''')
code('''display(pd.DataFrame(prep['thresholds']))
display(pd.Series(prep['coverage_full_train'], name='Cobertura de tokens (%)'))
print('Pendiente log-log (rangos 100–10,000):', prep['zipf_slope_rank_100_10000'])
figura('zipf.png')
''')
md('''Zipf se observa aproximadamente en el tramo central; la pendiente cercana a -1 no implica cumplimiento exacto en todo el rango. Los umbrales se calculan sobre los 22M tokens antes de eliminar OOV; la cobertura top-k se calcula sobre todo train de WikiText. Son denominadores distintos y están etiquetados.''')
code('''# Implementación completa de construcción del corpus, conteos e IDs en el repositorio.
from src.preprocessing import prepare
print(inspect.getsource(prepare))
''')
md('''## 3. Investigación: PyTorch, gensim, CBOW, skip-gram y GloVe
'''+file('docs/investigacion.md').split('## Tablas de embeddings en PyTorch',1)[1].split('## Normalización y corpus',1)[0])
code('''# Demostración de offsets y de mean/sum/max en EmbeddingBag.
W = torch.tensor([[0.,0.], [1.,2.], [3.,4.], [5.,6.]])
ids = torch.tensor([1, 2, 3])
offsets = torch.tensor([0, 2])  # documentos: [1,2] y [3]
for mode in ['mean', 'sum', 'max']:
    bag = nn.EmbeddingBag.from_pretrained(W, freeze=True, mode=mode, padding_idx=0)
    print(mode, bag(ids, offsets))
frozen = nn.Embedding.from_pretrained(W, freeze=True, padding_idx=0)
trainable = nn.Embedding.from_pretrained(W, freeze=False, padding_idx=0)
print('freeze=True requires_grad:', frozen.weight.requires_grad)
print('freeze=False requires_grad:', trainable.weight.requires_grad)
''')
md('''## 4. SGNS propio: dos tablas y negative sampling
Para centro c y contexto o, se minimiza `-log sigmoid(W_c·W'_o) - Σ log sigmoid(-W_c·W'_n)`. Los negativos siguen f^0.75; cuando un negativo coincide con o se descarta, como en word2vec. Se usan gradientes dispersos y SGD sobre la suma del minibatch. Esto conserva la escala de actualizaciones por par, aunque no el orden online de gensim. La pérdida reportada se divide entre pares. W se exporta como representación final.''')
code(inspect.getsource(SGNS))
md('''### Subsampling y generación de pares
Se usa `min(1,(sqrt(f/t)+1)t/f)`, variante de word2vec/gensim. Primero se filtran OOV, luego se descartan apariciones y finalmente se construyen ventanas fijas. Cada epoch tiene una muestra nueva. La función de pares no cruza límites de líneas; produce todos los pares, sin un recorte arbitrario de pasos.''')
code(inspect.getsource(keep_prob)+'\n'+inspect.getsource(generate_pairs.py_func).replace('@njit(cache=True)', '@njit(cache=False)'))
code('''from src.sgns import train, subsample, CONFIGS, pair_count
print(inspect.getsource(subsample))
print(inspect.getsource(train))
''')
code('''iterations = pd.read_csv(ROOT / 'results/iterations.csv')
display(iterations)
selection = resultado('selection.json')
print('Selección:', selection)
sgns_run = resultado('sgns_' + selection['best'] + '.json')
for record in sgns_run['records']:
    print('Epoch', record['epoch'], 'pérdida', record['loss'],
          'pares antes/después', record['pairs_before_subsampling'], record['pairs_after_subsampling'],
          'tiempo train', record['train_seconds'], 'GPU pico bytes', record['gpu_peak_bytes'])
display(pd.DataFrame(sgns_run['records'][-1]['subsampling']).T)
figura('curvas_embeddings.png')
figura('analogias_vs_tokens.png')
''')
md('''La selección sólo usa WS353 y questions-words: score = Spearman + accuracy total. El modelo principal es el mejor entre configuraciones con >=20M tokens conocidos; se identifica además el mejor de 100 dimensiones. SimLex no participa. La pérdida SGNS no optimiza directamente accuracy de analogías; cambiar k cambia su escala y no permite comparaciones directas de pérdida.''')
code('''# Métricas, vecinos fijos y tiempos de CADA configuración y CADA epoch.
for path in sorted((ROOT / 'results').glob('sgns_*.json')):
    run = json.loads(path.read_text())
    print('\\nConfiguración:', run['config'])
    display(pd.DataFrame([{'epoch':r['epoch'], 'loss':r['loss'],
        'semantic':r['analogies']['accuracy']['semantic'],
        'syntactic':r['analogies']['accuracy']['syntactic'],
        'total':r['analogies']['accuracy']['total'],
        'WS353':r['wordsim']['spearman'], 'train_seconds':r['train_seconds'],
        'evaluation_seconds':r['evaluation_seconds'], 'gpu_peak_bytes':r['gpu_peak_bytes']}
        for r in run['records']]))
    for r in run['records']:
        print('Epoch', r['epoch'], 'vecinos:')
        display(pd.DataFrame(r['neighbors']).T)
''')
md('''### Referencia gensim y GloVe
La referencia gensim emplea `sg=1`, `hs=0`, `shrink_windows=False`, el mismo corpus/vocabulario y valores de dimensión, ventana, negativos, sample, LR y epochs del mejor SGNS. Usa cuatro workers y actualizaciones online, por lo que no se afirma equivalencia numérica de la optimización. La pérdida de gensim es un acumulador aproximado y no está en la escala de la media SGNS.

GloVe se carga sin entrenamiento adicional. Los pesos originales se conservan para evaluar; fine-tuning de clasificación usa copias. El tiempo/hardware original del archivo exacto 100d no aparece inequívocamente en las fuentes consultadas y se deja N/D.''')
code('''print((ROOT / 'scripts/reference_and_final.py').read_text())
ref = resultado('gensim_training.json')
for record in ref['records']:
    print('Epoch', record['epoch'], 'analogías:', record['analogies']['accuracy'],
          'WS353:', record['wordsim'], 'segundos:', record['train_seconds'])
    display(pd.DataFrame(record['neighbors']).T)
''')
md('''## 5. Analogías propias, 3CosAdd, 3CosMul y paralelismo
`analogia(a,b,c,k)` resuelve a:b :: c:?. Normaliza individualmente a,b,c, normaliza la suma y excluye las tres palabras. Esto coincide con `most_similar(positive=[b,c], negative=[a])`. No se confunde con normalizar sólo una suma de vectores crudos.''')
code(inspect.getsource(unit)+'\n'+inspect.getsource(analogia))
code('''from gensim.models import KeyedVectors
from src.evaluation import benchmark, individual, similarity
sgns = KeyedVectors.load(str(ROOT / 'models' / (selection['best'] + '.kv'))) if (ROOT / 'models' / (selection['best'] + '.kv')).exists() else KeyedVectors.load_word2vec_format(str(ROOT / 'models/mejor_sgns.txt'))
print(analogia(sgns, 'man', 'woman', 'king', 5))
print(sgns.most_similar(positive=['woman','king'], negative=['man'], topn=5))
final = resultado('embedding_final.json')
for name, result in final.items():
    assert result['individual']['matches_gensim_all']
    print(name, 'coincidencia verificada con gensim:', result['individual']['matches_gensim_all'])
''')
code('''# 21 consultas × 3 modelos, con top-5/scores, rango correcto y sin exclusión.
for name, result in final.items():
    print('\\nModelo:', name)
    rows = result['individual']['queries']
    display(pd.DataFrame(rows)[['type','a','b','c','expected','top5','expected_rank',
                               'expected_cosine','without_exclusion','difference_cosine','relative_residual']])
''')
md('''### Benchmark compartido y análisis de desplazamientos
Se toman las 30,000 palabras más frecuentes de WikiText presentes en SGNS, gensim y GloVe. Sólo se evalúan preguntas cuyas cuatro palabras pertenecen a esa lista. Tanto preguntas como candidatos coinciden entre modelos. Cobertura = preguntas evaluadas / 19,544. Accuracy es sobre las evaluadas; no cuenta OOV como errores.

La guía remite a una sección 5.3 ausente. Para cumplir el entregable de paralelismo se incluyen cos(v_b-v_a, v_d-v_c) y residuo relativo de esos desplazamientos sobre vectores crudos. Un coseno cercano a 1 implica dirección parecida, pero no magnitud igual; un ranking correcto tampoco demuestra igualdad algebraica.''')
code('''print(inspect.getsource(benchmark))
for name, result in final.items():
    print(name, 'paralelismo medio:', result['individual']['parallelism_mean'])
    for method, ev in result['analogies'].items():
        print(method, ev['accuracy'], 'cobertura:', ev['evaluated'], '/', ev['total_questions'])
display(pd.read_csv(ROOT / 'results/benchmark_categories.csv'))
figura('analogias_categorias.png')
figura('tsne.png')
print('Palabras t-SNE:', len(resultado('tsne_words.json')))
''')
md('''t-SNE usa 500 vectores unitarios, perplexity=30 y semilla fija. Los grupos se definen antes de proyectar con países, números y categorías WordNet. Se observan agrupamientos parciales y solapamientos; polisemia y distorsión 2D impiden tratar la figura como prueba cuantitativa. La selección exacta y coordenadas se conservan en `tsne_words.json`.''')
md('''## 6. Clasificación de AG News
108,000 train y 12,000 validación, estratificados; 7,600 test oficial. Se prueban TF-IDF + regresión logística, EmbeddingBag aleatorio entrenable, y SGNS/GloVe congelados y con fine-tuning. Se añade gensim como control para completar su F1 downstream.

El vocabulario de tarea se construye sólo con train, frecuencia mínima 2. Cada fuente conserva cobertura propia; los tokens sin vector se mapean a UNK. Se distinguen OOV del embedding y UNK de la tabla de tarea. Arquitectura MLP 128/ReLU/dropout 0.2/4 clases; 10 epochs, batch 512, Adam y SparseAdam a 0.003. Random usa la dimensión SGNS; otras fuentes usan su dimensión nativa. El input cambia si el SGNS elegido tiene dimensión distinta de 100, una limitación reportada.

Se seleccionan epochs y variantes por F1 macro de validación. Sólo al terminar toda selección se abre test; cada ganador por inicialización y fracción se evalúa una vez. La curva de datos utiliza un modelo diferente por fracción, no ajustes repetidos de un modelo mirando test. El vocabulario de tarea permanece fijo en fracciones menores y usa texto no etiquetado del train completo; TF-IDF se ajusta sólo con la fracción correspondiente.''')
code(inspect.getsource(NewsClassifier)+'\n'+inspect.getsource(collate))
code('''from src.classification import train_network, run
print(inspect.getsource(train_network))
print(inspect.getsource(run))
''')
code('''display(pd.Series(resultado('classification_protocol.json')))
for name, stats in resultado('classification_oov.json').items():
    print(name)
    display(pd.DataFrame(stats).T)
# Todas las variantes, seleccionadas y no seleccionadas, con métricas/tiempos/parámetros.
clf_rows = []
for path in sorted((ROOT / 'results').glob('clf_*.json')):
    r = json.loads(path.read_text())
    clf_rows.append({'name':r['name'], 'n_train':r['n_train'],
        **r['validation'], 'parameters':r['total_parameters'],
        'trainable':r['trainable_parameters'], 'seconds':r['training_seconds']})
display(pd.DataFrame(clf_rows))
figura('curvas_clasificacion.png')
''')
md('''TF-IDF usa unigramas, min_df=2, sublinear_tf, máximo 60,000 atributos y C=4 en regresión logística. Se registran curvas mediante seis bloques de diez iteraciones L-BFGS con warm_start. La pérdida es entropía cruzada sin el término regularizador; no son epochs de entrenamiento SGD.''')
code('''display(pd.read_csv(ROOT / 'results/classification_test.csv'))
figura('matrices_confusion.png')
figura('f1_vs_datos.png')
# SimLex-999 reservado: evaluación final después de la selección intrínseca.
display(pd.DataFrame({name:{'WordSim rho':r['wordsim']['spearman'],
    'WordSim n':r['wordsim']['evaluated'], 'SimLex rho':r['simlex']['spearman'],
    'SimLex n':r['simlex']['evaluated']} for name,r in final.items()}).T)
''')
md('''## 7–8. Comparación, discusión y conclusiones
''')
if (ROOT/'docs/discusion_resultados.md').exists():
    md(file('docs/discusion_resultados.md'))
code('''clf = resultado('classification_final.json')
full = {r['initialization']:r for r in clf if r['fraction']==1}
small = {r['initialization']:r for r in clf if r['fraction']==.01}
oov = resultado('classification_oov.json')
comparison = []
for name, r in final.items():
    comparison.append({'modelo':name, 'tokens':6_000_000_000 if name=='GloVe' else sgns_run['corpus_tokens_known'],
        'vocabulario':r['vocab_size'], 'dim':r['dim'],
        'tiempo_s':None if name=='GloVe' else (ref if name=='gensim' else sgns_run)['training_seconds'],
        'hardware':'Original 100d: N/D' if name=='GloVe' else 'Apple M1 Pro, CPU',
        'Add_sem':r['analogies']['add']['accuracy']['semantic'],
        'Add_sint':r['analogies']['add']['accuracy']['syntactic'],
        'Add_total':r['analogies']['add']['accuracy']['total'],
        'Mul_sem':r['analogies']['mul']['accuracy']['semantic'],
        'Mul_sint':r['analogies']['mul']['accuracy']['syntactic'],
        'Mul_total':r['analogies']['mul']['accuracy']['total'],
        'WordSim':r['wordsim']['spearman'], 'SimLex':r['simlex']['spearman'],
        'paralelismo':r['individual']['parallelism_mean'],
        'AGNews_OOV_test_pct':oov[name]['test']['source_oov_percent'],
        'F1_test':full[name]['test']['f1_macro']})
display(pd.DataFrame(comparison))
''')
code('''# Respuestas basadas en resultados, sin asumir de antemano al ganador.
categories = final['SGNS']['analogies']['add']['categories']
covered = [r for r in categories if r['accuracy'] is not None]
strong = max(covered, key=lambda r:r['accuracy'])
weak = min(covered, key=lambda r:r['accuracy'])
print('Mejor/peor categoría SGNS:', strong['category'], strong['accuracy'], '/', weak['category'], weak['accuracy'])
print('Semántica y sintaxis:', final['SGNS']['analogies']['add']['accuracy'])
for name,r in final.items():
    q=r['individual']['queries'][0]
    print(name, 'king-man+woman:', q['top5'], '| sin exclusión:', q['without_exclusion'])
print('F1 al 100%:', {n:r['test']['f1_macro'] for n,r in full.items()})
print('F1 al 1%:', {n:r['test']['f1_macro'] for n,r in small.items()})
print('Variantes elegidas 100%:', {n:r['selected_run'] for n,r in full.items()})
print('Variantes elegidas 1%:', {n:r['selected_run'] for n,r in small.items()})
''')
md('''**Aritmética.** Los resultados por categoría, rangos y cosenos muestran relaciones aproximadas, no igualdades. Sin exclusión, términos de consulta pueden quedar primeros. El coseno de diferencias y el residuo relativo permiten comprobar que paralelismo y coincidencia algebraica son cosas distintas.

**Escala de datos.** La curva con 25/50/100% mantiene d=100, w=2 y k=3. Permite observar la tendencia de cantidad/composición del corpus, pero no separar causalmente el método del dominio: GloVe usa aproximadamente 6,000M tokens frente a 21M de WikiText. No se atribuye toda la brecha a una sola causa.

**Implementación.** La tendencia SGNS/gensim debe compararse con sus métricas, no con sus pérdidas de distinta escala. Gensim aplica actualizaciones online en varios workers; PyTorch suma un minibatch, baraja pares y genera otras muestras. Igualar hiperparámetros no hace idénticas esas trayectorias.

**Dimensión, ventana y negativos.** La tabla de iteraciones responde con resultados observados: subir dimensión no necesariamente ayuda con pocas epochs. Las ventanas cortas suelen capturar contexto sintáctico y las amplias contexto temático, pero las categorías medidas son la evidencia de este experimento. Más negativos modifica objetivo y costo; una pérdida mayor puede acompañar mejores representaciones.

**Clasificación.** La comparación incluye todas las variantes en validación, y en test sólo ganadores. Los resultados impresos muestran si superan random o TF-IDF, y qué variante convino al 100% y al 1%. Congelar reduce parámetros entrenables; fine-tuning permite adaptar al dominio. Esto no garantiza de antemano cuál obtiene mejor F1.

**Límites.** Una semilla, tres epochs, prefijos con sesgo temático, cobertura intrínseca diferente en WordSim/SimLex y posibles dimensiones distintas entre fuentes. No se presentan intervalos de confianza ni se generaliza una tendencia como ley causal. t-SNE es exploratorio. Tiempo/hardware exactos del GloVe 100d: no disponibles en fuentes consultadas.''')
md('''## Validación de la implementación y entregables
Se verifican pares sin cruzar líneas, la fórmula de subsampling, la pérdida inicial y sus gradientes, equivalencia 3CosAdd con gensim y agregación por offsets. Los resultados completos están en JSON/CSV; los vectores finales en `models/mejor_sgns.txt`, formato word2vec de texto. El informe resume la ejecución en cinco páginas.

Referencias principales: [PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html), [EmbeddingBag](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.sparse.EmbeddingBag.html), [gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html), [KeyedVectors](https://radimrehurek.com/gensim/models/keyedvectors.html), [Mikolov et al.](https://arxiv.org/abs/1310.4546), [GloVe](https://nlp.stanford.edu/projects/glove/), [Pennington et al.](https://aclanthology.org/D14-1162/).''')
code('''subprocess.run([sys.executable, 'scripts/verify_pipeline.py'], check=True)
print('Informe:', ROOT / 'output/pdf/Laboratorio_7_Informe.pdf')
print('Vectores:', ROOT / 'models/mejor_sgns.txt')
''')
md('''### Integridad de la ejecución
La rueda oficial gensim 4.4.0 emitió mensajes Cython our_dot_float. No se modificó Word2Vec ni se ocultaron sus avisos. La auditoría de resultados en results/audit.json verifica vectores finitos/no nulos, 21 epochs SGNS, preguntas compartidas y tamaños de matrices de confusión. Versiones exactas: requirements-lock.txt.''')
if (ROOT/'results/delivery.json').exists():
    url=json.loads((ROOT/'results/delivery.json').read_text())['repository']
    md('## Repositorio de entrega\n\n['+url+']('+url+')\n\nRepositorio público. Informe sencillo en Word: `output/docx/Laboratorio_7_Informe.docx`. Vectores: models/mejor_sgns.txt.gz.')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':platform.python_version() if False else '3.12'}})
nbf.write(nb,ROOT/'Laboratorio_7_NLP_Embeddings.ipynb')
print('Notebook creado:',len(cells),'celdas')
