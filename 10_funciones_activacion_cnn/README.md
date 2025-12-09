# Comparación de Funciones de Activación en Redes Convolucionales (CNN)


## **Propósito**

Este notbook busca **analizar y comparar** cómo distintas **funciones de activación** afectan el **aprendizaje y desempeño** de una red neuronal convolucional (CNN) profunda entrenada sobre el conjunto de datos **MNIST** (dígitos escritos a mano).

A través de un experimento controlado, se entrenan **tres modelos idénticos** —con la misma arquitectura, datos y optimizador— variando únicamente la **función de activación interna**:

- **ReLU (Rectified Linear Unit)**  
- **Tanh (Tangente hiperbólica)**  
- **Sigmoid**

---

## **Contexto teórico**

### ¿Qué es una función de activación?

Una **función de activación** define cómo una neurona transforma su entrada en salida.  
Su objetivo es **introducir no linealidad**, permitiendo que la red neuronal aprenda **relaciones complejas** entre los datos.

Sin activaciones, una red sería solo una combinación lineal de sus entradas, incapaz de aprender patrones no lineales (por ejemplo, distinguir curvas, letras o formas).

---

### Tipos comparados en este experimento

| Activación | Ecuación | Rango | Ventajas | Desventajas |
|-------------|-----------|--------|------------|--------------|
| **Sigmoid** | f(x) = 1 / (1 + exp(-x)) | (0, 1) | Salida en forma de probabilidad | Se satura (gradientes ≈ 0) |
| **Tanh** | f(x) = tanh(x) | (-1, 1) | Centra las salidas en 0 | También puede saturar |
| **ReLU** | f(x) = max(0, x) | [0, ∞) | Rápida, no satura en positivos | Puede “matar” neuronas negativas |

Cada una impacta de forma distinta la propagación del gradiente y la velocidad de convergencia.

---

## **Estructura del experimento**

1. **Dataset:**  
   Se utiliza **MNIST**, un conjunto con 70,000 imágenes en escala de grises (28×28 px) de dígitos del 0 al 9.  
   - 60,000 imágenes para entrenamiento.  
   - 10,000 imágenes para prueba.

2. **Preprocesamiento:**  
   Las imágenes se convierten a tensores y se normalizan al rango **[-1, 1]**, adecuado para activaciones simétricas como *tanh*.

3. **Arquitectura de la red (CNN profunda):**
   - 3 bloques convolucionales con filtros de 3×3 y *MaxPooling* (reduce tamaño espacial).
   - Capas densas finales para clasificación.
   - Activación intercambiable (ReLU, Tanh o Sigmoid).

4. **Entrenamiento:**
   - Optimización con **Adam** (lr=0.001).
   - Pérdida: **CrossEntropyLoss** (clasificación multiclase).
   - Entrenamiento por 3 épocas sobre todos los datos.

5. **Evaluación:**
   - Se mide el **tiempo total de entrenamiento**.
   - Se calcula la **precisión (accuracy)** sobre el conjunto de prueba.
   - Los resultados se guardan y se presentan en una tabla comparativa.

---

## **Flujo general del código**

1️. Cargar librerías y configurar PyTorch  
2️. Descargar y normalizar el dataset MNIST  
3️. Definir el modelo CNN (build_deep_cnn_model)  
4️. Implementar funciones de entrenamiento y evaluación  
5️. Ejecutar el bucle de comparación (ReLU, Tanh, Sigmoid)  
6️. Mostrar resultados finales  

---

## **Resultados esperados**

| Activación | Tiempo (s) | Precisión (%) | Comportamiento |
|-------------|-------------|----------------|----------------|
| **ReLU** | Rápido | 98–99% | Entrenamiento estable, buen gradiente |
| **Tanh** | Medio | 97–99% | Ligeramente más lento, pero preciso |
| **Sigmoid** | Similar | ~10% | Falla: gradientes saturados (aprendizaje nulo) |

> Este comportamiento ilustra el fenómeno del **desvanecimiento del gradiente**, donde funciones como *Sigmoid* pierden información en redes profundas.

---

## **Conclusiones**

- La **ReLU** domina en redes profundas: simple, eficiente y evita saturación.  
- **Tanh** funciona bien en redes pequeñas o con datos centrados en cero.  
- **Sigmoid**, aunque útil en salidas binarias, no es adecuada para capas internas profundas.  
- Este tipo de experimentos son esenciales para entender **cómo la elección de activación puede determinar el éxito o fracaso del entrenamiento.**

---

**Autor:**  
Saúl Rovelo López  
Universidad Autónoma Metropolitana – Unidad Cuajimalpa  
Curso: *Aprendizaje de Máquina aplicado a Teoría de Gráficas*
