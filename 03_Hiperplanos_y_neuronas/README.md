# Hiperplanos, Neuronas y Regresión Logística  

Este notebook desarrolla paso a paso los fundamentos matemáticos y computacionales necesarios para entender cómo funcionan los **hiperplanos**, la **regresión logística**, el **gradiente ascendente** y el papel de una **neurona** en un modelo de clasificación binaria.  


### **1. Producto punto → Implementación directa**
El producto punto se implementa *sin librerías*, usando bucles anidados:

```python
z = 0.0
for j in range(n):
    z += beta[j] * X[i][j]
```

Esto permite visualizar cómo la neurona combina características sin depender de frameworks externos.

---

### **2. Función sigmoide → Función explícita**
La sigmoide, definida matemáticamente como  
$$
\sigma(z)=\frac{1}{1+e^{-z}},
$$
se implementa de manera directa:

```python
def sigmoid(z):
    return 1 / (1 + math.exp(-z))
```

Esto permite observar su comportamiento numérico y cómo transforma el valor lineal en una probabilidad.

---

### **3. Probabilidades del modelo → Uso del producto punto + sigmoide**
Cada probabilidad se calcula así:

```python
z = beta · X
p = sigmoid(z)
```

Implementado explícitamente en:

```python
p[i] = sigmoid(z)
```

Esto conecta la regresión logística con el comportamiento de una neurona individual.

---

### **4. Verosimilitud y log-verosimilitud → Cálculo estructurado**
El notebook no oculta la fórmula:  
se construyen las expresiones de manera iterativa para que el estudiante vea:

- cómo surge el término  
  $$ y \log(p) $$
- cómo aparece  
  $$ (1-y)\log(1-p) $$

y cómo se suma todo en:

```python
LL += y[i] * math.log(p[i]) + (1 - y[i]) * math.log(1 - p[i])
```

---

### **5. Gradiente ascendente → Implementado manualmente**
El gradiente teórico  
$$
\frac{\partial LL}{\partial \beta_j}=\sum (y_i - p_i)x_{ij}
$$
se implementa con bucles:

```python
grad[j] = 0.0
for i in range(m):
    grad[j] += (Y[i] - p[i]) * X[i][j]
```

Actualización:

```python
beta[j] += lr * grad[j]
```

Esto permite entender *exactamente* cómo se optimizan los parámetros.

---

### **6. Hiperplano → Visualización**
El hiperplano de decisión  
$$
\beta \cdot X = 0
$$
se convierte en una recta en 2D:

```python
x2 = -(b0 + b1*x1)/b2
```

El notebook grafica:

- puntos de clase 0  
- puntos de clase 1  
- la recta del hiperplano

permitiendo ver cómo la regresión logística separa las clases.

---

### **7. Neuronas y activaciones → Conexión final**
El notebook muestra:

- parte lineal: `z = beta · X`
- parte no lineal: `sigmoid(z)`
- salida binaria mediante umbral

lo que permite visualizar cómo una neurona es un clasificador lineal.

---

# Conclusión

Este notebook integra de manera clara y progresiva los fundamentos matemáticos con su implementación computacional, mostrando cómo conceptos como el producto punto, la sigmoide, la verosimilitud y el gradiente ascendente se articulan para construir un modelo de **regresión logística** completamente funcional.  

Al implementar cada paso manualmente —sin depender de librerías automáticas— se facilita una comprensión profunda de cómo una neurona procesa información, cómo se entrena mediante gradientes y cómo su función lineal induce un hiperplano capaz de separar clases en el espacio de características.  

Finalmente, las visualizaciones y el vínculo con PyTorch permiten conectar estos modelos básicos con arquitecturas más avanzadas como los perceptrones multicapa y las redes neuronales profundas, evidenciando que los algoritmos modernos se sustentan en estas ideas fundamentales de estadística, optimización y geometría.
