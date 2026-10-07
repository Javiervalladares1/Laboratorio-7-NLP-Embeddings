# Laboratorio 7: NLP end-to-end y embeddings

**Javier Valladares, 23045 · Ian Cumes, 23236**  
CC3092 - Deep Learning y Sistemas Inteligentes

Repositorio público: [https://github.com/Javiervalladares1/Laboratorio-7-NLP-Embeddings](https://github.com/Javiervalladares1/Laboratorio-7-NLP-Embeddings).

La entrega incluye SGNS propio en PyTorch, referencia gensim, GloVe 100d, evaluación intrínseca y clasificación AG News. Las métricas son resultados ejecutados; no hay valores simulados.

## Entregables

- `Laboratorio_7_NLP_Embeddings.ipynb`: notebook comentado, con resultados y visualizaciones.
- `output/pdf/Laboratorio_7_Informe.pdf`: informe de máximo cinco páginas.
- `output/docx/Laboratorio_7_Informe.docx`: informe en Word con lenguaje sencillo, texto, tablas y gráficos en negro.
- `models/mejor_sgns.txt`: vectores del mejor SGNS, formato word2vec de texto.
- `results/`: configuraciones, métricas por epoch, vecinos, analogías propias, categorías, clasificación, cobertura y selección.
- `figures/`: Zipf, curvas, t-SNE, benchmark y matrices de confusión.
- `src/`: implementación reusable del pipeline.
- `docs/investigacion.md`: investigación, referencias y decisiones ampliadas.

## Instalación y ejecución

Python 3.12 recomendado. Desde la raíz:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
python scripts/run_all.py
```

Para abrir el notebook:

```bash
python -m pip install jupyterlab
jupyter lab Laboratorio_7_NLP_Embeddings.ipynb
```

El notebook usa `REENTRENAR=False` de forma predeterminada y muestra los resultados ya guardados. La lectura del mejor modelo funciona directamente con `models/mejor_sgns.txt`, sin descargar otra vez GloVe. Para reproducir desde cero, activar `REENTRENAR=True` o ejecutar `scripts/run_all.py`. Se descargan datasets y GloVe a `data/`. Se requiere espacio para datos, modelos y pares temporales; la ejecución de referencia se hizo en M1 Pro con 16 GB, CPU de cuatro hilos.

Para regenerar el informe de Word desde los resultados guardados, sin reentrenar:

```bash
pip install -r requirements-documents.txt
python scripts/make_word_report.py
```

`run_all.py` reutiliza configuraciones completas y test ya sellado. Para un experimento independiente, usar una copia nueva del proyecto con `results/`, `models/` y `data/` reconstruidos, conservando la entrega original. No borrar selectivamente resultados de test para ajustar sobre ellos.

## Protocolo

- WikiText: 22M tokens normalizados de train; min_count=20 deja 21,336,925 tokens conocidos y 38,569 palabras. Ventanas no cruzan líneas.
- Siete configuraciones SGNS, tres epochs cada una: dimensión 50/100/300, corpus 25/50/100%, ventana 2/5 y negativos 3/5.
- Modelo principal elegido sólo entre configuraciones con >=20M tokens conocidos; criterio: Spearman WordSim-353 + accuracy total de analogías. SimLex reservado.
- Referencia gensim: mismo corpus, vocabulario y parámetros equivalentes. Diferencias de orden, minibatch y paralelismo están documentadas.
- Analogías: 30,000 candidatos comunes, mismos denominadores/preguntas; 3CosAdd y 3CosMul, 14 categorías y 21 consultas propias de siete tipos.
- AG News: split estratificado 108k/12k; test oficial 7,600. Épocas y frozen/FT seleccionados en validación; test una vez por ganador y fracción (1/10/50/100%). Se añade gensim downstream.
- La referencia a sección 5.3 del PDF no contiene una sección real; se incluye paralelismo de diferencias y residuo relativo para cubrir el entregable.

Los tiempos del propio SGNS incluyen preparación de pares por epoch y excluyen evaluación; se registra evaluación por separado. El entrenamiento usa CPU y el pico GPU es cero. La memoria RAM del equipo no se presenta como memoria GPU. El tiempo/hardware original del archivo GloVe exacto 100d no está inequívocamente documentado en fuentes y se marca N/D.

## Verificación

```bash
python scripts/verify_pipeline.py
```

Comprueba límites de ventanas, pérdida y gradientes SGNS, subsampling, coincidencia 3CosAdd con gensim y offsets de EmbeddingBag. El notebook se ejecuta de extremo a extremo sobre resultados guardados y el PDF se revisa por renderización.

## Archivos grandes y fuentes

`data/`, `.venv/` y checkpoints internos se omiten de Git. Los vectores finales se distribuyen como `models/mejor_sgns.txt.gz`. El notebook los descomprime automáticamente si no existe la versión `.txt`. También puede usarse `gzip -dk models/mejor_sgns.txt.gz`. Datos públicos: [WikiText](https://huggingface.co/datasets/Salesforce/wikitext), [AG News](https://huggingface.co/datasets/fancyzhx/ag_news). Modelo: [GloVe original](https://nlp.stanford.edu/projects/glove/). Documentación y papers enlazados en el notebook.
