# Investigación y decisiones

Autores: Javier Valladares (23045) e Ian Cumes (23236).

## Tablas de embeddings en PyTorch

`nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, sparse=False)` guarda una matriz V × d y devuelve las filas indicadas por IDs. `num_embeddings` incluye todas las filas necesarias, y `embedding_dim` fija d. `padding_idx` excluye esa fila de los gradientes; su inicialización predeterminada es cero. `sparse=True` produce gradientes dispersos, no una tabla dispersa: la matriz de pesos sigue siendo densa. SGD y SparseAdam admiten estos gradientes; Adam estándar no. `nn.Embedding.from_pretrained(weights, freeze=True)` carga una matriz existente y controla si sus pesos se actualizan. La tokenización y el orden de IDs deben coincidir con las filas de la matriz.

Fuente: [PyTorch, Embedding y from_pretrained](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html).

`nn.EmbeddingBag` agrega las filas de cada documento sin materializar un tensor completo con padding. Para entrada 1D, `offsets` contiene el índice inicial de cada documento. `mean` promedia, `sum` suma y `max` obtiene máximos por coordenada; este último no admite gradientes dispersos. Con `include_last_offset=True`, se agrega un offset final igual al número de tokens. Aquí se usa `mean`, `include_last_offset=False` y una fila de padding excluida. Un documento vacío se representa por UNK. El promedio pierde el orden y la longitud explícita del texto; el MLP recibe sólo d valores.

Fuente: [PyTorch, EmbeddingBag](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.sparse.EmbeddingBag.html).

## Skip-gram, CBOW y las dos matrices

CBOW predice la palabra central a partir de los contextos; skip-gram predice contextos a partir de la palabra central. SGNS usa una tabla de entrada W y una tabla de salida W': una palabra puede desempeñar ambos papeles, pero los parámetros cumplen funciones diferentes. Para un par positivo (c,o), se minimiza `-log σ(W_c · W'_o) - Σ log σ(-W_c · W'_n)`. Los negativos se muestrean con probabilidad proporcional a frecuencia^0.75. Se conserva W para las evaluaciones y la clasificación. W' sigue siendo necesaria durante el entrenamiento, aunque no se exporte como representación final.

Fuente: [Mikolov et al., 2013](https://arxiv.org/abs/1310.4546).

## Gensim

`Word2Vec` es la referencia: `sg=1` elige skip-gram; `vector_size`, `window`, `negative`, `sample`, `min_count`, `alpha`, `min_alpha`, `epochs` y `ns_exponent` controlan la dimensión, contexto, negativos, subsampling, vocabulario y optimización. `shrink_windows=False` fija la ventana para compararla con la implementación propia. `build_vocab_from_freq` recibe los mismos conteos filtrados y asegura un vocabulario idéntico. `train` recorre líneas tokenizadas y usa cuatro workers. La comparación conserva corpus, objetivo, dimensión, ventana, negativos, subsampling y calendario de tasa de aprendizaje; difieren el orden de actualización y el paralelismo: PyTorch suma gradientes por minibatch y gensim actualiza por pares con concurrencia. No se afirma identidad numérica.

`KeyedVectors` guarda vocabulario/vectores sin el estado del optimizador. `most_similar` resuelve vecinos y 3CosAdd; `most_similar_cosmul` implementa 3CosMul. `evaluate_word_analogies` y `evaluate_word_pairs` proporcionan evaluadores de referencia. El laboratorio usa además evaluadores propios para hacer explícitos candidatos, exclusiones, preguntas compartidas y cobertura. `api.load('glove-wiki-gigaword-100')` descarga el modelo preentrenado; `save_word2vec_format(..., binary=False)` exporta el mejor SGNS.

Fuentes: [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html), [KeyedVectors](https://radimrehurek.com/gensim/models/keyedvectors.html).

## Word2Vec y GloVe

Word2Vec aprende predictivamente con pares de contexto; GloVe ajusta una función de mínimos cuadrados ponderados sobre conteos globales de coocurrencia. El modelo descargado tiene 400,000 palabras, 100 dimensiones y procede de Wikipedia 2014 + Gigaword 5, con 6,000 millones de tokens y texto en minúsculas. No se entrena adicionalmente para la comparación intrínseca. El fine-tuning downstream crea copias independientes y no modifica el GloVe de referencia. Las fuentes consultadas no proporcionan un tiempo y hardware inequívocos para este archivo concreto de 100 dimensiones: se reportan como no disponibles, sin trasladar mediciones de 300 dimensiones.

Fuentes: [GloVe, modelos originales](https://nlp.stanford.edu/projects/glove/), [Pennington et al., 2014](https://aclanthology.org/D14-1162/).

## Normalización y corpus

Se aplican NFKC y minúsculas; el regex se restringe a letras ASCII del inglés (por ejemplo, café se fragmenta en caf) y dígitos; se separan puntuación, apóstrofos y guiones. Se conservan números literales y no se crean tokens `<num>`. Se eliminan etiquetas como `<unk>` y marcadores de WikiText `@-@`, `@,@`, `@.@`. No se eliminan stopwords ni se aplica stemming o lematización. Las contracciones se fragmentan (`can't` -> `can`, `t`), una decisión simple pero imperfecta, explicitada en las doce comparaciones con Treebank de NLTK. GloVe también es uncased, pero no se supone que sus reglas de tokenización sean idénticas: se mide cobertura y se usa un vocabulario compartido para analogías.

Se cuentan artículos por encabezados de nivel uno (`= título =`), líneas como filas del dataset y tokens después de normalización. El corpus de entrenamiento es un prefijo determinista de 22M tokens, truncando sólo su última línea, y se conserva la separación entre líneas en las ventanas. Tras `min_count=20`, se mantienen más de 20M tokens. El prefijo facilita reproducibilidad, pero puede introducir sesgo temático; el experimento de tamaño no aísla únicamente cantidad, pues también cambia contenido. Las fracciones menores son ablaciones, no sustituyen al modelo principal de al menos 20M.

La probabilidad de conservar una palabra es `min(1, (sqrt(f/t)+1)*t/f)`, la variante usada en word2vec/gensim, donde f es su frecuencia relativa. Se submuestrea antes de construir la ventana y se recalcula la muestra por epoch. Se reportan descarte teórico y observado, y el número exacto de pares antes/después.

## Aritmética y paralelismo (sección 5.3 ausente)

El PDF pide paralelismo y remite a una sección 5.3 que no contiene. Para hacerlo explícito, se calcula `cos(v_b-v_a, v_d-v_c)` y el residuo relativo `|| (v_b-v_a)-(v_d-v_c) || / (||v_b-v_a||+||v_d-v_c||)` en las 21 analogías propias (7 tipos). El coseno 1 sólo representa direcciones paralelas; no garantiza igual magnitud. El residuo 0 sí indica igualdad de esos dos desplazamientos. Se usan vectores crudos en este análisis. Para 3CosAdd, en cambio, se normaliza cada vector antes de sumar y se normaliza la consulta, conforme a `most_similar`. También se repite la búsqueda sin excluir las palabras de consulta.

## Selección y prevención de fuga

Todas las configuraciones tienen tres epochs y semilla 23045. La selección usa la suma `Spearman WordSim-353 + accuracy total de questions-words` con los mismos candidatos; esta ponderación se fija antes de ver SimLex. El modelo principal se elige entre los entrenados con al menos 20M tokens conocidos; se identifica también el mejor de dimensión 100. SimLex se abre sólo en la evaluación final.

AG News se divide en 108,000 ejemplos de train y 12,000 de validación, estratificados. La tabla de la tarea usa palabras de frecuencia >=2 en train, nunca validación/test. Se mantienen la arquitectura, el MLP de 128 unidades, dropout 0.2, optimizadores, semilla y diez epochs. Los OOV de cada fuente se convierten en UNK; se reporta también el filtrado adicional del vocabulario de la tarea. A fracciones pequeñas se conserva el vocabulario de entrenamiento completo, sin usar las etiquetas de las noticias omitidas. TF-IDF aprende su vocabulario sólo con la fracción correspondiente. Esto debe considerarse al interpretar el escenario de 1%.

Se seleccionan epoch y variante congelada/fine-tuning por F1 macro de validación. Sólo después se evalúa una vez el checkpoint ganador de cada inicialización y fracción en test. La curva F1 vs datos requiere un ganador distinto por fracción; no se reevalúa ni se ajusta ningún modelo usando test. Se añade gensim como control downstream para completar la tabla solicitada. No se comparan las escalas de pérdida de gensim y PyTorch ni pérdidas SGNS con diferente número de negativos como si fueran el mismo objetivo.

## Auditoría del entorno

La rueda oficial gensim 4.4.0 emitió 35 mensajes de Cython `our_dot_float` durante la referencia, sin excepción Python capturable. No se modificó la implementación de Word2Vec ni se suprimieron sus mensajes. Se comprobaron pérdida finita, vectores finitos/no nulos, mejora progresiva de los benchmarks y coincidencia de los evaluadores. Las versiones exactas están en requirements-lock.txt y los resultados de integridad en results/audit.json. Este aviso de la biblioteca es una limitación del entorno de ejecución, y debe conservarse al interpretar o reproducir la referencia.
