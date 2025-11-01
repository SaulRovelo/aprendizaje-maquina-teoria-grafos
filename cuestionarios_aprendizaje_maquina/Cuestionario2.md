
# 🧩 Cuestionario 2 de Repaso

### 1️⃣ En una neurona con función lineal z y función de activación f(z), ¿qué representa geométricamente que z=0?

✅ **Respuesta:**  
Un **hiperplano que separa dos clases** en el espacio de entrada.

💡 **Explicación:**  
El plano definido por z=0 actúa como **frontera de decisión**, dividiendo las clases según el signo de z.

---

### 2️⃣ Si y = g(x) y z = f(y), ¿cuál de las siguientes expresiones representa correctamente la derivada dz/dx según la regla de la cadena?

✅ **Respuesta:**  
`dz/dx = dz/dy * dy/dx`

💡 **Explicación:**  
La **regla de la cadena** multiplica derivadas parciales intermedias,  
expresando cómo cambia z con respecto a x a través de y.

---

### 3️⃣ ¿Cuál de las siguientes operaciones construye una función indicadora en el intervalo [a,b] usando funciones escalón H(x)?

✅ **Respuesta:**  
`Resta: H(x−a) − H(x−b)`

💡 **Explicación:**  
`H(x−a)` se activa en a, `H(x−b)` se apaga en b;  
su resta vale 1 solo dentro del intervalo [a,b].

---

### 4️⃣ ¿Qué establece el Teorema de Aproximación Universal?

✅ **Respuesta:**  
Una **red con una sola capa oculta** y suficientes neuronas puede **aproximar cualquier función continua** en un intervalo.

💡 **Explicación:**  
El teorema (Hornik y Cybenko, 1989) muestra que las redes neuronales pueden aproximar cualquier función continua,  
si tienen suficientes neuronas y una función de activación no lineal.

---

### 5️⃣ En PyTorch, si una variable z depende de otra variable x, ¿cómo se obtiene la derivada dz/dx?

✅ **Respuesta:**  
Llamando a `z.backward()` y consultando el valor en `x.grad`.

💡 **Explicación:**  
`backward()` realiza **backpropagation automática** y almacena la derivada de z respecto a x en `x.grad`.

---

### 6️⃣ ¿Qué ocurre en PyTorch si intentas sumar dos tensores que están en diferentes dispositivos (uno en CPU y otro en GPU)?

✅ **Respuesta:**  
Se genera un **error**, porque los tensores deben estar **en el mismo dispositivo**.

💡 **Explicación:**  
PyTorch **no convierte automáticamente** entre CPU y GPU.  
Es necesario mover manualmente los tensores con `.to('cuda')` o `.to('cpu')`.

---

### 7️⃣ ¿A qué función se aproxima σ(wx) (la sigmoide) cuando w → ∞?

✅ **Respuesta:**  
A una **función escalón (Heaviside)**.

💡 **Explicación:**  
Cuando w crece, la sigmoide se comporta como un **escalón binario**:  
1 si x>0, 0 si x<0.

---

### 8️⃣ ¿Cuál es una limitación fundamental de usar una sola neurona para clasificación?

✅ **Respuesta:**  
No puede **separar regiones no linealmente separables**.

💡 **Explicación:**  
Una neurona define un **hiperplano lineal**, por lo que no puede resolver problemas como XOR.

---
V
### 9️⃣ ¿Cuál es la ventaja principal de realizar operaciones en GPU con PyTorch?

✅ **Respuesta:**  
Permite **procesar grandes cantidades de datos más rápido** que en CPU.

💡 **Explicación:**  
Las GPU tienen miles de núcleos para operaciones en paralelo,  
acelerando el entrenamiento de modelos de redes neuronales.

---
