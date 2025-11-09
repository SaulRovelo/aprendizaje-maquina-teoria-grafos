
# Gradiente Ascendente sin Vectores

## Introducción

Este notebook presenta la implementación completa del algoritmo de **gradiente ascendente** utilizando **bucles explícitos**, sin operaciones vectorizadas de NumPy.

El propósito es mostrar de manera transparente cómo se calculan las **probabilidades**, el **gradiente** y la **actualización de los parámetros** de un modelo logístico.  
Cada paso del proceso se expresa de forma elemental, con el objetivo de reforzar la comprensión del algoritmo desde su nivel más básico.

---

## 1. Conjunto de datos y propósito

El código trabaja con un conjunto de datos sencillo:

```python
X = np.array([
    [0, 0, 1, 0],
    [1, 1, 1, 1],
    [0, 0, 0, 1]
])
Y = np.array([1, 0, 1])
```

- Cada **fila** de `X` representa un ejemplo.  
- Cada **columna** representa una característica (*feature*).  
- `Y` contiene las etiquetas reales (1 para clase positiva, 0 para clase negativa).

Este formato permite observar con claridad cómo los parámetros del modelo se ajustan de manera progresiva.

---

## 2. Función sigmoide

La **función sigmoide** se usa para transformar combinaciones lineales en valores entre 0 y 1, que se interpretan como probabilidades.

Definición conceptual:

```
sigma(z) = 1 / (1 + exp(-z))
```

### En el código

```python
def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))
```

- La entrada `z` es un número real.  
- La salida representa la probabilidad de que `Y = 1` dado el valor `z`.  
- Esta implementación es **escalar**, sin vectorización: se evalúa ejemplo por ejemplo dentro de los bucles del algoritmo.

---

## 3. Cálculo de probabilidades

Esta función calcula las **probabilidades predichas** para cada ejemplo aplicando la función sigmoide sobre la combinación lineal de parámetros y características.

```python
def calcular_probabilidades(X, beta):
    m = len(X)
    n = len(beta)
    p = [0.0 for _ in range(m)]

    for i in range(m):
        z = 0.0
        for j in range(n):
            z += beta[j] * X[i][j]
        p[i] = sigmoid(z)
    return p
```

**Proceso:**
1. Recorre cada ejemplo `i`.  
2. Calcula la suma ponderada `z = β₀x₀ + β₁x₁ + ... + βₙxₙ`.  
3. Aplica `sigmoid(z)` para obtener la probabilidad predicha `p[i]`.

Esta etapa equivale a la operación `p = sigmoid(X @ beta)` en versiones vectorizadas.

---

## 4. Cálculo del gradiente

El gradiente mide la dirección de máximo incremento de la función objetivo.  
En este caso, se acumula la contribución de cada ejemplo al gradiente de cada parámetro `β_j`.

```python
def calcular_gradiente(X, Y, p):
    n = len(X[0])
    m = len(X)
    grad = [0.0 for _ in range(n)]

    for j in range(n):
        suma = 0.0
        for i in range(m):
            suma += (Y[i] - p[i]) * X[i][j]
        grad[j] = suma
    return grad
```

**Interpretación:**
- `(Y[i] - p[i])` representa el error de predicción para el ejemplo `i`.  
- Cada término `X[i][j]` pondera ese error por el valor de la característica correspondiente.  
- La suma total `grad[j]` acumula la influencia de cada ejemplo sobre el parámetro `β_j`.

---

## 5. Norma del gradiente

Para definir una **condición de paro**, se calcula la norma Euclidiana del gradiente.  
Cuando esta norma se vuelve muy pequeña, el algoritmo se considera convergente.

```python
def norma_vector(v):
    suma = 0.0
    for i in range(len(v)):
        suma += v[i] * v[i]
    return math.sqrt(suma)
```

Esto equivale a `||v||₂ = sqrt(sum(v[i]^2))`.

---

## 6. Actualización de los parámetros

Una vez calculado el gradiente, los parámetros `β` se actualizan en la dirección del ascenso:

```python
def actualizar_beta(beta, grad, eta):
    for j in range(len(beta)):
        beta[j] += eta * grad[j]
    return beta
```

Donde:
- `eta` es la **tasa de aprendizaje**, que controla la magnitud del paso.  
- Cada parámetro `β_j` se incrementa proporcionalmente al valor del gradiente correspondiente.

---

## 7. Algoritmo principal de gradiente ascendente

Integra todas las funciones anteriores y ejecuta el proceso iterativo de aprendizaje.

```python
def gradiente_ascendente(X, Y, eta=0.01, max_iter=1000, tol=0.0001):
    n = len(X[0])
    beta = [0.0 for _ in range(n)]

    for iteration in range(max_iter):
        p = calcular_probabilidades(X, beta)
        grad = calcular_gradiente(X, Y, p)
        grad_norm = norma_vector(grad)

        if grad_norm < tol:
            print(f"Convergió en {iteration} iteraciones")
            break

        beta = actualizar_beta(beta, grad, eta)

    return beta
```

### Descripción del flujo:
1. **Inicializa** los parámetros `β` en cero.  
2. **Calcula** las probabilidades predichas `p`.  
3. **Evalúa** el gradiente en base al error `(Y - p)`.  
4. **Actualiza** los parámetros con `β = β + η * grad`.  
5. **Detiene** el proceso si `||grad|| < tol` o se alcanzan las iteraciones máximas.

---

## 8. Ejecución y resultados

El entrenamiento se ejecuta llamando al algoritmo principal:

```python
betas_finales = gradiente_ascendente(X, Y, eta=0.01, max_iter=5000, tol=1e-5)
print("Parámetros finales aprendidos (beta):")
print(betas_finales)
```

El resultado muestra los valores ajustados de los parámetros `β`, que indican la influencia de cada característica en la probabilidad de que `Y = 1`.

---

## 9. Observaciones

- El algoritmo sin vectorización es menos eficiente, pero didácticamente más claro.  
- Cada función implementa un paso fundamental del método de optimización.  
- El uso de bucles explícitos permite seguir el cálculo de cada componente del gradiente.

Este enfoque sienta las bases para comprender el funcionamiento de versiones vectorizadas o implementaciones en librerías como PyTorch o TensorFlow.

---

## Referencias

- Robles, I. *Aprendizaje de Máquina Aplicado a Teoría de Gráficas*. UAM-Cuajimalpa, 2025.  
- Bishop, C. M. *Pattern Recognition and Machine Learning*. Springer, 2006.  
- Murphy, K. P. *Machine Learning: A Probabilistic Perspective*. MIT Press, 2012.  
- Notas y diapositivas del curso (tema: Gradiente ascendente y verosimilitud).
