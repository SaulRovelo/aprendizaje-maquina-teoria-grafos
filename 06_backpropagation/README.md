# Backpropagation – Implementación Computacional en Python

Este notebook muestra cómo se aplica el algoritmo de backpropagation utilizando el motor de derivadas automáticas de PyTorch. El enfoque es completamente práctico: cada concepto se explica a través de código, mostrando cómo se construye el grafo computacional y cómo se propagan los gradientes hacia las variables de entrada.

---

## Tensores con gradiente y construcción del grafo

El notebook comienza creando tensores con `requires_grad=True`.  
Esto indica a PyTorch que debe rastrear todas las operaciones realizadas con ellos:

```python
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
v = torch.tensor(3.0, requires_grad=True)
```

Cada operación posterior forma parte del grafo computacional sobre el cual se aplicará backpropagation.  
En términos matemáticos, cualquier función compuesta del tipo:

$$
w = f(a, b, v)
$$

comenzará a registrar las dependencias necesarias para aplicar la regla de la cadena.

---

## Composición de operaciones en código

Para observar el funcionamiento del gradiente, el notebook define operaciones simples:

```python
u = v + b
w = a * u
```

En notación matemática:  $u = v + b, \qquad w = a \cdot u
$

Estas líneas construyen una función compuesta. PyTorch registra automáticamente esta estructura, permitiendo que posteriormente la llamada a `w.backward()` recorra el grafo y calcule:

$$
\frac{\partial w}{\partial a}, \quad
\frac{\partial w}{\partial v}, \quad
\frac{\partial w}{\partial b}
$$

sin escribir derivadas explícitas.

---

## Aplicación directa de backpropagation

La instrucción central es:

```python
w.backward()
```

Esto activa el retrocálculo de gradientes.  
Los resultados pueden inspeccionarse con:

```python
a.grad
v.grad
b.grad
```

El notebook muestra cómo estos valores corresponden exactamente a las derivadas que resultan de aplicar la regla de la cadena sobre la composición definida por el código.

---

## Acumulación y reinicio de gradientes

Otro concepto aplicado explícitamente es la acumulación de gradientes.  
PyTorch no reemplaza `.grad`, sino que lo suma en cada llamada sucesiva a `backward()`.

El notebook lo demuestra ejecutando backpropagation varias veces y consultando los gradientes acumulados.

Para reiniciarlos se utiliza:

```python
a.grad.zero_()
b.grad.zero_()
v.grad.zero_()
```

Esto permite observar claramente cómo cambian los resultados cuando los gradientes se acumulan y cuando se reinician.

---

## Funciones no lineales dentro del grafo

El notebook incluye funciones como:

- `torch.sin`
- `torch.exp`
- `torch.log`
- potencias y combinaciones no lineales

Por ejemplo:

```python
y = torch.sin(a * v) + torch.exp(b)
z = y**2
z.backward()
```

En notación matemática: $ y = \sin(av) + e^{b}, \qquad z = y^{2}$

Las derivadas de estas funciones se calculan automáticamente dentro del grafo sin necesidad de implementarlas manualmente.  
Esto muestra cómo PyTorch extiende la regla de la cadena a expresiones más complejas.

---

## Ejemplo compuesto

El notebook finaliza con una función más elaborada que combina operaciones lineales y no lineales.  
Esto permite observar un grafo computacional mayor y cómo backpropagation recorre todas las rutas aplicando:

$$
\frac{d}{dx} f(g(h(x)))
$$

a través de las dependencias construidas en el código.

---

## Conclusión

El notebook aplica los conceptos de backpropagation directamente en código mediante:

- la creación de tensores diferenciables,
- la construcción de funciones compuestas,
- el uso de operaciones lineales y no lineales,
- la ejecución de `backward()`,
- y la observación explícita de los gradientes producidos.

Con ello, se muestra cómo los principios matemáticos del cálculo de derivadas y la regla de la cadena se traducen en instrucciones de PyTorch que automatizan por completo el proceso de entrenamiento de redes neuronales.

--- 
**Autor:**  
Saúl Rovelo López  
Universidad Autónoma Metropolitana – Unidad Cuajimalpa  
Curso: *Aprendizaje de Máquina aplicado a Teoría de Gráficas*
