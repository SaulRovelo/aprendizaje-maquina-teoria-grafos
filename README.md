# Aprendizaje de Máquina Aplicado a Teoría de Grafos

## Descripción

Este repositorio reúne notebooks y materiales orientados al estudio progresivo de los fundamentos matemáticos y computacionales del aprendizaje de máquina, con énfasis en su aplicación a modelos neuronales y a estructuras basadas en grafos.  
Incluye contenidos que abarcan técnicas de optimización, gradiente ascendente, neuronas logísticas, teorema de aproximación universal, backpropagation, redes densas y redes convolucionales aplicadas a MNIST, así como métricas de evaluación de modelos, con implementaciones prácticas en Python y PyTorch.

## Objetivo del proyecto

El propósito de este repositorio es proporcionar una base sólida para comprender el funcionamiento interno de los modelos de aprendizaje de máquina y su relación con las estructuras matemáticas que los sustentan.  
Asimismo, se pretende mostrar cómo estos modelos pueden extenderse a escenarios donde los datos ya no son vectores aislados, sino entidades relacionadas entre sí, como ocurre en las gráficas. El proyecto busca dejar un conjunto de notebooks reproducibles que documenten el comportamiento de distintas arquitecturas, faciliten su análisis comparativo y sirvan como base para nuevos experimentos.

## Contenido 

**Fundamentos de optimización y gradiente**  
Se presenta el gradiente ascendente en sus distintas variantes, su relación con la verosimilitud y el entrenamiento de modelos logísticos. Se incluyen representaciones intuitivas y derivaciones explícitas.

**Neuronas, hiperplanos y funciones de activación**  
Se analiza el comportamiento geométrico de hiperplanos, el papel del umbral en clasificación, la función sigmoide y sus limitaciones. También se estudian funciones de activación utilizadas en arquitecturas más avanzadas.

**Teorema de aproximación universal**  
Se estudia su interpretación práctica y sus implicaciones para el diseño de redes neuronales, incluyendo ejemplos guiados que ilustran cómo una red puede aproximar funciones continuas.

**Backpropagation y entrenamiento computacional**  
Se detalla el cálculo de gradientes en redes neuronales, la regla de la cadena y la ejecución automática de derivadas en PyTorch. Se revisa la importancia del diseño de la función de pérdida y su estabilidad numérica.

**Redes neuronales aplicadas a visión por computadora**  
Se estudian CNNs desde sus componentes fundamentales: convoluciones, filtros, padding, stride y pooling. Se explican sus implicaciones en tareas reales, como la clasificación de imágenes (ej. MNIST).

**Representación y análisis de grafos**  
Incluye las nociones básicas de gráficas dirigidas y no dirigidas, así como problemas de invariancia a permutaciones y estrategias tradicionales de generación de embeddings.

**Redes neuronales gráficas y GCN**  
Se profundiza en el modelo GCN, la idea de paso de mensajes, los métodos de agregación, la normalización simétrica y la construcción de capas convolucionales sobre grafos.  
Se ejemplifica su uso con PyTorch Geometric y datasets como CORA.

## Tecnologías utilizadas

El proyecto se desarrolla en un entorno basado en Python e integra herramientas diseñadas para el modelado numérico y el aprendizaje de máquina. Entre las tecnologías empleadas destacan:

- Python como lenguaje principal para la experimentación.
- PyTorch para la definición, entrenamiento y análisis de modelos neuronales.
- PyTorch Geometric para la implementación de operaciones y arquitecturas sobre grafos.
- Jupyter Notebook como entorno interactivo para la ejecución de código y documentación de cada módulo.
- Bibliotecas científicas como NumPy, pandas, Matplotlib y SciPy para soporte numérico, análisis y visualización.

## Requisitos para ejecución

Para ejecutar los notebooks se requiere un entorno Python configurado con las dependencias incluidas en `requirements.txt`. Entre los componentes esenciales se encuentran:

- Python 3.12
- PyTorch 2.x con soporte CPU o GPU
- PyTorch Geometric y sus extensiones asociadas
- NumPy, Matplotlib, pandas y librerías científicas auxiliares
- Jupyter Notebook o JupyterLab para la interacción con los notebooks


## Enfoque académico y aplicaciones

El proyecto está orientado a estudiantes e investigadores interesados en comprender el funcionamiento interno de los modelos neuronales. El enfoque combina fundamentos teóricos, interpretación geométrica y experimentos reproducibles que facilitan el análisis de distintas arquitecturas y métodos de entrenamiento.  

Los conceptos abordados son aplicables en áreas como visión por computadora, análisis de redes sociales, bioinformática, sistemas de recomendación y, en general, en cualquier dominio donde los datos presenten relaciones estructuradas.


## Autoría

El contenido fue preparado y organizado por **Saúl Rovelo**, integrando notebooks, experimentos y material de apoyo orientado al análisis y comprensión de distintos modelos de aprendizaje de máquina y su aplicación en grafos.