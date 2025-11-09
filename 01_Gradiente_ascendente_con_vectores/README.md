# Gradiente Ascendente con Vectores

## Introducción

Este notebook desarrolla la aplicación práctica del gradiente ascendente
usando operaciones vectorizadas con NumPy, mostrando cómo los conceptos
de álgebra lineal se traducen en código.

El objetivo es comprender cómo se implementan:

-   La función sigmoide, que transforma combinaciones lineales en
    valores entre 0 y 1 (interpretables como probabilidades).
-   El gradiente ascendente, que ajusta los parámetros del modelo para
    maximizar una función objetivo (por ejemplo, la log-verosimilitud en
    regresión logística).

Todo el código se explica paso a paso, relacionando la teoría con su
implementación computacional.

------------------------------------------------------------------------

## 1. Operaciones con vectores y matrices

Antes de aplicar el gradiente ascendente, el notebook repasa operaciones
fundamentales con NumPy:

-   Vectores como arreglos 1D: `v = np.array([1, 2, 3])`.
-   Producto punto: `np.dot(v1, v2)`.
-   Producto cruz: `np.cross(v1, v2)`.
-   Multiplicación de matrices: `m3 = m1 @ m2`.
-   Transpuesta: `m3.T`.
-   Matrices de ceros y unos: `np.zeros((m, n))`, `np.ones((m, n))`.
-   Dimensiones: `mat.shape`.
-   Cambio de forma: `reshape`.
-   Apilamiento horizontal: `np.hstack`.

Estas herramientas permiten escribir actualizaciones como:

    beta <- beta + eta * X^T (y - p)

de forma directa y vectorizada, sin bucles explícitos.

------------------------------------------------------------------------

## 2. Función sigmoide

La función sigmoide se utiliza para mapear una combinación lineal de
características a un valor entre 0 y 1.

Definición:

    sigma(z) = 1 / (1 + exp(-z))

En el contexto de modelos de clasificación binaria:

-   `z = X @ beta` es la combinación lineal.
-   `p = sigma(z)` es la probabilidad estimada de la clase positiva.

### Implementación en el código

``` python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

Puntos clave:

-   La implementación es vectorizada: `z` puede ser un escalar, vector o
    matriz.
-   `p = sigmoid(X @ beta)` produce un vector de probabilidades, una por
    cada fila de `X`.

------------------------------------------------------------------------

## 3. Gradiente ascendente vectorizado

En modelos tipo regresión logística se busca maximizar una función
objetivo (por ejemplo, la log-verosimilitud). El gradiente ascendente
actualiza los parámetros en la dirección que incrementa dicha función.

Idea general:

    beta_nueva = beta_actual + eta * gradiente

Para el caso logístico, el gradiente tiene la forma:

    gradiente = X^T (y - p)

donde:

-   `X` es la matriz de características (m x n).
-   `y` es el vector columna de etiquetas reales (m x 1).
-   `p = sigmoid(X @ beta)` son las probabilidades predichas.
-   `X^T` es la transpuesta de `X`.
-   `eta` es la tasa de aprendizaje.

### Implementación en el código

``` python
def grad_asc(X, y, eta=0.1, epochs=1000):
    m, n = X.shape
    beta = np.zeros((n, 1))

    for _ in range(epochs):
        z = X @ beta          # combinación lineal
        p = sigmoid(z)        # probabilidades predichas
        grad = X.T @ (y - p)  # gradiente vectorizado
        beta = beta + eta * grad / m  # actualización de parámetros

    return beta
```

Relación con la teoría:

-   `X @ beta` implementa la combinación lineal.
-   `sigmoid(z)` implementa la función de activación probabilística.
-   `X.T @ (y - p)` corresponde al gradiente calculado de manera
    matricial.
-   La actualización `beta = beta + eta * grad / m` es la versión
    programática de la regla de gradiente ascendente.

------------------------------------------------------------------------

## 4. Aplicación del gradiente ascendente

En el notebook se define un conjunto de datos sintético:

``` python
Xc = np.array([
    [0, 0, 1, 0],
    [1, 1, 1, 1],
    [0, 0, 0, 1]
])

Y = np.array([1, 0, 1]).reshape((-1, 1))
```

Interpretación:

-   Cada fila de `Xc` es un ejemplo.
-   Cada columna es una característica.
-   `Y` contiene las etiquetas verdaderas: `1` (clase positiva), `0`
    (clase negativa).

Entrenamiento:

``` python
betas = grad_asc(Xc, Y, eta=0.1, epochs=5000)
print(betas)
```

Este llamado:

1.  Calcula iterativamente `z = Xc @ beta`.
2.  Aplica `p = sigmoid(z)`.
3.  Obtiene el gradiente `Xc.T @ (Y - p)`.
4.  Actualiza `beta` hasta que converge a un conjunto de parámetros que
    se ajustan a los datos dados.

------------------------------------------------------------------------

## 5. Interpretación del resultado

El vector `beta` aprendido indica la contribución de cada característica
a la predicción:

-   Componentes positivas de `beta` tienden a incrementar la
    probabilidad de clase positiva cuando la característica asociada
    está activa.
-   Componentes negativas la disminuyen.

Aunque el ejemplo es pequeño, el procedimiento es exactamente el mismo
que se utilizaría con conjuntos de datos más grandes y reales.

------------------------------------------------------------------------

## Referencias

-   Robles, I. *Aprendizaje de Máquina Aplicado a Teoría de Gráficas*.
    UAM-Cuajimalpa, 2025.\
-   Bishop, C. M. *Pattern Recognition and Machine Learning*. Springer,
    2006.\
-   Murphy, K. P. *Machine Learning: A Probabilistic Perspective*. MIT
    Press, 2012.\
-   Notas y diapositivas del curso (tema: Gradiente ascendente y
    verosimilitud).
