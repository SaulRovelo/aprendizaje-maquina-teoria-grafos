# 🧩 Cuestionario 1 de Repaso

### 1️⃣ ¿Cuándo es preferible usar el algoritmo del gradiente ascendente en lugar del método directo para obtener el valor de las β?

✅ **Respuesta:**  
Cuando el modelo tiene **funciones trascendentes**, como las exponenciales.

💡 **Explicación:**  
El método directo no puede resolver funciones con exponentes o sigmoides;  
el gradiente ascendente optimiza iterativamente hasta encontrar el mejor valor.

---

### 2️⃣ En un modelo probabilístico, ¿cómo se define la verosimilitud?

✅ **Respuesta:**  
Es la **función que mide la probabilidad de obtener los datos observados** dado un conjunto de parámetros del modelo.

💡 **Explicación:**  
La verosimilitud indica **qué tan probables son los datos** si los parámetros del modelo son correctos.

---

### 3️⃣ ¿Por qué como condición de paro en el algoritmo del gradiente ascendente se usa que la norma del gradiente tienda a cero?

✅ **Respuesta:**  
Porque implica que se ha alcanzado un **punto crítico**.

💡 **Explicación:**  
Cuando el gradiente se acerca a cero, la función deja de cambiar:  
se ha alcanzado un **máximo o mínimo**.

---

### 4️⃣ ¿Por qué se aplica logaritmo a la verosimilitud?

✅ **Respuesta:**  
Porque **permite derivar más fácilmente** al aplicar el gradiente ascendente.

💡 **Explicación:**  
El logaritmo **convierte productos en sumas**, lo que simplifica las derivadas  
y evita errores numéricos por números demasiado pequeños.

---

### 5️⃣ ¿Qué sucede si el valor de η (tasa de aprendizaje) es demasiado grande?

✅ **Respuesta:**  
El algoritmo puede **oscilar sin converger** a una solución.

💡 **Explicación:**  
Una tasa de aprendizaje muy alta hace que los pasos sean demasiado grandes,  
haciendo que el algoritmo “rebote” y no alcance el máximo.

---

### 6️⃣ ¿Cuál es el resultado de `np.hstack([a,b])` con `a=[[1],[2]]` y `b=[[3],[4]]`?

✅ **Respuesta:**  
`[[1, 3], [2, 4]]`

💡 **Explicación:**  
`np.hstack()` concatena **por columnas**,  
formando una matriz de **2×2** al unir los vectores verticalmente.

---

### 7️⃣ En el algoritmo de máxima verosimilitud, ¿qué se asume sobre los vectores aleatorios X^(i)?

✅ **Respuesta:**  
Que son **condicionalmente independientes**.

💡 **Explicación:**  
Cada muestra depende únicamente de su propio **X^(i)** y de los parámetros del modelo,  
no de otras muestras.

---

### 8️⃣ ¿Cuál de los siguientes programas multiplica correctamente el producto punto entre dos arreglos NumPy `a` y `b`?

✅ **Respuesta:**  
`a.dot(b)`

💡 **Explicación:**  
`a.dot(b)` o `np.dot(a,b)` calculan el **producto punto**,  
es decir, multiplican y suman los elementos correspondientes.

---

### 9️⃣ ¿Por qué se usa el modelo de regresión logística en lugar de una distribución de probabilidad conjunta discreta?

✅ **Respuesta:**  
Porque **guardar la distribución completa es intratable** en términos de memoria.

💡 **Explicación:**  
Una distribución conjunta requiere almacenar **2ⁿ combinaciones**,  
lo cual es imposible para valores grandes de *n*;  
la regresión logística es **más eficiente y práctica**.

---

