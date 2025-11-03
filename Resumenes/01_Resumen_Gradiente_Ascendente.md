# 📘 Entendiendo el Gradiente Ascendente en Regresión Logística

Este resumen explica en lenguaje natural dos implementaciones del algoritmo de **gradiente ascendente** para regresión logística:  
una usando **vectores (NumPy)** y otra con **bucles explícitos (sin vectores)**.  
Está diseñado para que puedas estudiarlo, explicarlo o grabarlo fácilmente.

---

## 🧠 ¿Qué es una neurona logística?

Una neurona logística toma varias entradas \( x_i \), las combina linealmente usando pesos \( w_i \), agrega un sesgo \( b \),  
y aplica la función sigmoide para convertir eso en una **probabilidad**:

$$
z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b, \quad \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

El objetivo es ajustar los pesos \( \beta = [b, w_1, \dots, w_n] \) para que \( \hat{y} \) se aproxime a la etiqueta real \( y \).

---

## 📗 01 – Gradiente Ascendente **con vectores (NumPy)**

### 🔧 ¿Qué hace?
Implementa regresión logística con gradiente ascendente usando álgebra lineal (matrices y vectores con NumPy).

### 🪜 Pasos clave:

1. **Inicializa \( \beta \) en ceros.**
2. **Agrega columna de unos a \( X \)** para representar el sesgo.
3. En cada iteración:
   - Calcula \( \hat{y} = \sigma(X \cdot \beta) \)
   - Calcula el gradiente:  
     $$ \nabla = X^T \cdot (Y - \hat{y}) $$
   - Evalúa la norma del gradiente para saber si hay que seguir.
   - Actualiza los pesos:
     $$ \beta := \beta + \eta \cdot \nabla $$

### 💬 ¿Cómo explicarlo?
> El modelo predice con una combinación lineal y la sigmoide.  
> Compara esa predicción con el valor real.  
> El gradiente le dice hacia dónde ajustar los pesos para mejorar.  
> Y la actualización da un paso en esa dirección. Repite hasta que converge.

---

## 📙 02 – Gradiente Ascendente **sin vectores (con bucles)**

### 🔧 ¿Qué hace?
Implementa lo mismo, pero **paso a paso** usando listas y ciclos for para entender mejor el funcionamiento interno.

### 🪜 Componentes clave:

- `calcular_probabilidades`: evalúa la sigmoide manualmente.
- `calcular_gradiente`: acumula los errores manualmente para cada peso.
- `norma_vector`: mide la magnitud del gradiente.
- `actualizar_beta`: aplica la fórmula de actualización a mano.

### 💬 ¿Cómo explicarlo?
> Aquí hacemos todo sin librerías mágicas: sumamos, multiplicamos y derivamos manualmente.  
> Eso nos obliga a ver cómo cada error se propaga y ajusta los pesos.  
> Es más lento, pero excelente para aprender desde cero cómo aprende un modelo.

---

## 🔁 ¿Qué comparten ambos?

| Etapa | Propósito |
|-------|-----------|
| Predicción \( \hat{y} = \sigma(X \cdot \beta) \) | Obtener probabilidad de clase positiva |
| Error \( Y - \hat{y} \) | Comparar con etiquetas reales |
| Gradiente \( \nabla = X^T (Y - \hat{y}) \) | Saber en qué dirección mejorar |
| Actualización \( \beta = \beta + \eta \cdot \nabla \) | Ajustar los pesos |

---

## ✅ Conclusión

Ambos programas muestran cómo una neurona logística puede **aprender por sí sola** a separar clases ajustando sus pesos.  
Ya sea con NumPy o con bucles, el corazón del algoritmo es el mismo:

> Predecir → Comparar → Calcular error → Derivar → Actualizar.

Ese ciclo es el alma del **gradiente ascendente** 💡
