# Teorema de Aproximación Universal – Implementación Computacional

Este notebook implementa de forma práctica los elementos necesarios para mostrar cómo una red neuronal sencilla puede aproximar funciones continuas. El objetivo central es evidenciar, mediante código, que ciertas combinaciones de sigmoides permiten construir aproximaciones arbitrariamente precisas.

El enfoque de la libreta es computacional: cada bloque de código ilustra directamente un componente funcional utilizado en la construcción de la aproximación.

## Función sigmoide y su papel en el código

El notebook inicia implementando la función sigmoide:

```python
def sigmoid(x, w=1, b=0):
    return 1 / (1 + np.exp(-(w*x + b)))
```

En el código, la sigmoide se utiliza como un “suavizador” capaz de generar transiciones controladas. Al incrementar el parámetro `w`, la curva se hace más abrupta y se acerca a un escalón. Este comportamiento se aprovecha más adelante para construir funciones indicadoras y regiones acotadas.

Las gráficas iniciales permiten observar cómo los distintos valores de `w` modifican su forma.

## Construcción de funciones indicadoras mediante sigmoides

Para crear intervalos donde la función “se activa” o “se apaga”, el notebook utiliza la diferencia entre dos sigmoides:

```python
def indicator(x, a, b, w=50):
    return sigmoid(x, w, -w*a) - sigmoid(x, w, -w*b)
```

Esta operación produce una función que vale aproximadamente 1 en el intervalo [a, b] y ≈0 fuera de él.  
El parámetro `w` controla qué tan abrupto es el borde del intervalo.

El código grafica estos indicadores para verificar que efectivamente generan ventanas acotadas.

## Funciones barra (bump functions)

Las funciones barra se construyen asignando una altura específica a cada intervalo activado. En el código:

```python
def bump(x, a, delta, alpha, w=50):
    return alpha * indicator(x, a, a+delta, w)
```

La lógica es:

- Se activa la región [a, a + δ].
- Se multiplica por una constante `alpha`.
- La salida tiene forma de “barra” con altura definida.

Estas barras permiten representar valores puntuales de una función continua alrededor de los datos observados.

## Aproximación de la función objetivo

Con las barras definidas, el notebook construye una aproximación sumando varias de ellas. La estructura típica del código es:

```python
F = np.zeros_like(x)

for (a, alpha) in zip(xs, ys):
    F += bump(x, a, delta, alpha)
```

Donde:

- `xs` son los puntos de entrada.
- `ys` son los valores reales de la función objetivo.
- Cada barra representa localmente el comportamiento de la función.
- La suma de todas las barras produce una aproximación global.

El valor de `delta` determina qué tan concentrada es la contribución de cada punto.

## Visualización comparativa

El notebook muestra en una misma gráfica:

- La función real que se desea aproximar.
- La suma de barras que constituye la aproximación.
- La diferencia o error visual.

El contraste permite verificar cómo la aproximación mejora al ajustar `w`, `delta` o el número de puntos.

## Conclusión sobre el funcionamiento del código

El notebook demuestra computacionalmente que:

1. La sigmoide puede controlar regiones activas del dominio.
2. Restar sigmoides permite construir intervalos bien definidos.
3. Multiplicar esos intervalos por constantes produce bloques con altura específica.
4. La suma de estos bloques permite aproximar funciones continuas.
5. Esta construcción es equivalente a una red neuronal con una capa oculta.

El código no prueba formalmente el teorema, sino que muestra su mecanismo operativo mediante programación y visualización.

--- 
**Autor:**  
Saúl Rovelo López  
Universidad Autónoma Metropolitana – Unidad Cuajimalpa  
Curso: *Aprendizaje de Máquina aplicado a Teoría de Gráficas*
