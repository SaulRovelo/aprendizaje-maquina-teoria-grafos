# Neurona Logística OR en PyTorch — Explicación del Código

## Introducción

El notebook implementa una neurona logística que aprende a clasificar la función OR.  
Todo el proceso —construcción del dataset, definición del modelo, entrenamiento y evaluación— se realiza con PyTorch utilizando tensores, una capa lineal y la activación sigmoidal.

---

## Dataset OR

En código se define un conjunto pequeño de datos que representan la tabla OR:

```
X: [[0,0],
    [0,1],
    [1,0],
    [1,1]]

Y: [0,1,1,1]
```

Estos valores se convierten a tensores de PyTorch con tipo `float32`.  
Esto permite que PyTorch pueda calcular gradientes más adelante.

El propósito del dataset es simple: enseñarle al modelo a aproximar la función OR mediante aprendizaje automático en lugar de reglas explícitas.

---

## Construcción de la neurona logística

En el código se crea un modelo con PyTorch que representa una neurona binaria:

- Tiene **dos pesos**: uno para X1 y otro para X2.
- Tiene **un sesgo (bias)**.
- Aplica un cálculo lineal:  
  `z = w1 * x1 + w2 * x2 + b`
- Usa la activación sigmoide para convertir ese valor en una probabilidad entre 0 y 1.

En términos de PyTorch, esto suele implementarse con:

```python
self.linear = nn.Linear(2, 1)
self.sigmoid = nn.Sigmoid()
```

La función `forward()` del modelo indica el flujo de datos:

1. Entrada → capa lineal  
2. Capa lineal → sigmoide  
3. Sigmoide → salida final

Con esto, cada ejemplo genera una probabilidad estimada de ser clase 1.

---

## Función de pérdida

El notebook utiliza **Binary Cross Entropy (BCE)**.  
PyTorch calcula internamente:

```
-loss = Y*log(pred) + (1-Y)*log(1-pred)
```

La pérdida mide qué tan bien las probabilidades generadas coinciden con las etiquetas reales.

Valores importantes:

- pérdida alta → el modelo falla en las predicciones.  
- pérdida baja → las predicciones se acercan a los valores correctos.

---

## Ciclo de entrenamiento

El código usa un **loop de epochs** donde en cada iteración se realiza:

### 1. Forward pass
El modelo recibe X y produce una predicción:

```
pred = modelo(X)
```

### 2. Cálculo de la pérdida
```
loss = criterio(pred, Y)
```

### 3. Backpropagation
PyTorch calcula automáticamente los gradientes con:

```
loss.backward()
```

Esto llena:
```
w.grad
b.grad
```

con los valores necesarios para mover los pesos en la dirección correcta.

### 4. Actualización de parámetros
Con:
```
optimizer.step()
```

se ajustan los pesos y el bias utilizando los gradientes.

### 5. Limpieza de gradientes
Antes de pasar a la siguiente iteración:

```
optimizer.zero_grad()
```

si no se hace esto, los gradientes se acumulan de manera incorrecta.

---

## Aprendizaje real: comportamiento del modelo

Durante el entrenamiento, la neurona aprende los pesos correctos para replicar la regla OR.  
El modelo ajusta:

- `w1` y `w2` para reflejar que **cada entrada individual puede activar la salida**.
- `b` para ajustar el umbral.

El entrenamiento hace que:

- para entradas (0,0) → la salida tienda a 0  
- para (0,1), (1,0) y (1,1) → la salida tienda a 1  

El modelo generaliza la regla, sin que se la demos explícitamente.

---

## Evaluación del modelo

Después del entrenamiento se evalúan dos cosas:

- **Predicción numérica (probabilidad)**  
- **Predicción binaria (0 o 1) con un umbral**  

Cuando el modelo ha aprendido, se observa algo como:

```
Input: [0,0] → Prob ~0.02 → Clase 0
Input: [0,1] → Prob ~0.98 → Clase 1
Input: [1,0] → Prob ~0.97 → Clase 1
Input: [1,1] → Prob ~0.99 → Clase 1
```

Lo anterior indica que el modelo **ya reprodujo correctamente la función OR**.

---

**Autor:**  
Saúl Rovelo López  
Universidad Autónoma Metropolitana – Unidad Cuajimalpa  
Curso: *Aprendizaje de Máquina aplicado a Teoría de Gráficas*
