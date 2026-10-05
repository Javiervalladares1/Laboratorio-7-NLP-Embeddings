# Discusión basada en la ejecución

El modelo principal seleccionado fue **d100_w5**, epoch 3; supera el mínimo de 20 millones de tokens conocidos.

## ¿Se cumple la aritmética vectorial?

Se cumple parcialmente como recuperación de vecinos. SGNS tiene accuracy 3CosAdd 23.42%; su mejor categoría es gram6-nationality-adjective (59.89%) y su peor categoría cubierta es currency (0.93%). No implica que todos los desplazamientos sean exactos.

**SGNS:** semánticas 16.61%, sintácticas 26.53%. En man:woman :: king:queen, queen queda en rango 28, con coseno 0.639; el top-1 excluyendo es consort, y sin excluir es king. El coseno medio entre diferencias en las 21 consultas es 0.392.

**gensim:** semánticas 14.61%, sintácticas 25.94%. En man:woman :: king:queen, queen queda en rango 39, con coseno 0.607; el top-1 excluyendo es marriage, y sin excluir es king. El coseno medio entre diferencias en las 21 consultas es 0.381.

**GloVe:** semánticas 61.37%, sintácticas 67.77%. En man:woman :: king:queen, queen queda en rango 1, con coseno 0.770; el top-1 excluyendo es queen, y sin excluir es king. El coseno medio entre diferencias en las 21 consultas es 0.619.

## ¿Es una igualdad?

No. 3CosAdd usa vectores normalizados y un ranking; no exige residuo cero sobre vectores crudos. Al permitir palabras de consulta, la salida puede cambiar a una de ellas. Esto revela que parte de la cercanía procede de los términos presentes en la suma. El coseno entre diferencias mide dirección, y el residuo relativo mide discrepancia de dirección y magnitud; ninguna de las consultas acertadas constituye por sí sola una igualdad algebraica.

## Tamaño del corpus y distancia a GloVe

- 5.33M tokens, dimensión 100, w=2/k=3: 0.84% de analogías.
- 10.67M tokens, dimensión 100, w=2/k=3: 2.37% de analogías.
- 21.34M tokens, dimensión 100, w=2/k=3: 11.83% de analogías.

GloVe alcanza 65.76% con 6,000M tokens, unas 281.2 veces nuestro corpus. El aumento dentro de la ablación apoya un efecto favorable de más datos, pero los prefijos también añaden temas y cobertura. El SGNS principal puede tener otros hiperparámetros que esa ablación. No se puede repartir causalmente la brecha entre cantidad y método sin entrenar ambos métodos sobre exactamente los mismos datos y presupuesto.

## Implementación SGNS frente a gensim

La accuracy total es 23.42% en PyTorch y 22.40% en gensim; WordSim rho es 0.650 y 0.638, respectivamente. Se igualaron corpus, vocabulario y parámetros del objetivo. PyTorch suma gradientes de un minibatch; gensim actualiza por pares con cuatro workers. También difieren orden y muestras aleatorias. La coincidencia de la función de analogías y las pruebas de pérdida/gradientes verifica componentes matemáticos; no demuestra identidad de sus trayectorias de optimización.

## Dimensión, ventana y negativos

- d=50, w=2, k=3: total 11.46%; semánticas 9.36%; sintácticas 12.42%; WordSim 0.546.
- d=100, w=2, k=3: total 11.83%; semánticas 9.30%; sintácticas 12.98%; WordSim 0.551.
- d=300, w=2, k=3: total 10.89%; semánticas 8.80%; sintácticas 11.85%; WordSim 0.539.
- d=100, w=5, k=3: total 23.42%; semánticas 16.61%; sintácticas 26.53%; WordSim 0.650.
- d=100, w=2, k=5: total 15.28%; semánticas 11.62%; sintácticas 16.95%; WordSim 0.562.

Al ampliar w=2 a w=5, la mejora semántica observada es 7.31 puntos porcentuales y la sintáctica 13.55. Una ventana amplia aporta coocurrencias temáticas y más pares por epoch; el presupuesto computacional no queda constante. Una dimensión mayor no garantiza mejoras con sólo tres epochs. El número de negativos afecta objetivo, escala de pérdida y tiempo; no se ordenan modelos por pérdidas de distintos k.

## Clasificación: preentrenados, aleatorios y TF-IDF

### Con 100% de etiquetas disponibles
- TFIDF: F1 macro test 0.9114; variante elegida en validación: TFIDF_p100.
- random: F1 macro test 0.9174; variante elegida en validación: random_finetune_p100.
- SGNS: F1 macro test 0.9109; variante elegida en validación: SGNS_finetune_p100.
- gensim: F1 macro test 0.9117; variante elegida en validación: gensim_finetune_p100.
- GloVe: F1 macro test 0.9178; variante elegida en validación: GloVe_finetune_p100.
### Con 1% de etiquetas disponibles
- TFIDF: F1 macro test 0.8381; variante elegida en validación: TFIDF_p1.
- random: F1 macro test 0.7712; variante elegida en validación: random_finetune_p1.
- SGNS: F1 macro test 0.8426; variante elegida en validación: SGNS_finetune_p1.
- gensim: F1 macro test 0.8481; variante elegida en validación: gensim_finetune_p1.
- GloVe: F1 macro test 0.8644; variante elegida en validación: GloVe_finetune_p1.

Para SGNS con 100%, F1 de validación congelado=0.8777, fine-tuning=0.9057; convino fine-tuning según ese criterio. El test no se usó para tomar la decisión.

Para SGNS con 1%, F1 de validación congelado=0.8291, fine-tuning=0.8483; convino fine-tuning según ese criterio. El test no se usó para tomar la decisión.

Para GloVe con 100%, F1 de validación congelado=0.8992, fine-tuning=0.9152; convino fine-tuning según ese criterio. El test no se usó para tomar la decisión.

Para GloVe con 1%, F1 de validación congelado=0.8597, fine-tuning=0.8665; convino fine-tuning según ese criterio. El test no se usó para tomar la decisión.

Para gensim con 100%, F1 de validación congelado=0.8754, fine-tuning=0.9061; convino fine-tuning según ese criterio. El test no se usó para tomar la decisión.

Para gensim con 1%, F1 de validación congelado=0.8303, fine-tuning=0.8484; convino fine-tuning según ese criterio. El test no se usó para tomar la decisión.

Ganador observado al 100%: GloVe; al 1%: GloVe. Los embeddings preentrenados no se presumen superiores: sus resultados se comparan explícitamente con random y TF-IDF. El vocabulario fijo usa texto no etiquetado de todo train en las fracciones pequeñas; TF-IDF se ajusta sólo con su fracción. No hay intervalos de confianza y las diferencias pequeñas no prueban superioridad estadística.

## Límites

Una semilla, tres epochs de embeddings, prefijos con sesgo temático y cobertura de pares WordSim/SimLex distinta por fuente. Las analogías sí usan preguntas/candidatos idénticos. La normalización ASCII fragmenta algunos nombres y contracciones. EmbeddingBag pierde orden de palabras. t-SNE es exploratorio. Tiempo/hardware original del archivo GloVe 100d: N/D, no se atribuyen datos de otro archivo. Todas las cifras proceden de los JSON ejecutados; no se usan métricas simuladas.
