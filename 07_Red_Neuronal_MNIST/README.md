# Red Neuronal para Reconocimiento de Imágenes (MNIST 0 y 1)

Este notebook implementa una **red neuronal simple en PyTorch** para reconocer los dígitos **0 y 1** del conjunto **MNIST**, un clásico dataset que contiene imágenes de números escritos a mano. Se muestra cómo preparar los datos, definir el modelo, entrenarlo y evaluar su desempeño.

---

## Conceptos generales

### Tensores

En PyTorch, toda la información —imágenes, etiquetas, pesos y gradientes— se representa como **tensores**.  
Un tensor es una estructura parecida a un arreglo de Numpy, pero optimizada para operaciones en GPU, lo que permite acelerar el entrenamiento.  

Por ejemplo, una imagen en escala de grises de 28x28 píxeles se representa como un tensor con forma:  
```
(1, 28, 28)
```
donde:
- `1` indica el canal (grises),  
- `28x28` corresponde al tamaño de la imagen.

---

### Máscara de datos

El conjunto MNIST contiene imágenes de los dígitos del **0 al 9**, pero este notebook se enfoca en un **problema binario**: reconocer solo los dígitos **0 y 1**.  

Para lograrlo, se aplica una **máscara booleana**, que filtra los datos y conserva únicamente las muestras donde la etiqueta es 0 o 1:

```python
mascara = (targets == 0) | (targets == 1)
```

Esto simplifica el problema de clasificación, reduciéndolo a dos clases posibles.

---

### Características (X) y Etiquetas (Y)

- **Características (X):** son las imágenes de entrada, es decir, los píxeles del número escrito.  
- **Etiquetas (Y):** son las salidas esperadas o respuestas correctas del modelo (`0` o `1`).

Ambos se convierten a tipo **float** (`.float()`) para que puedan ser procesados correctamente por PyTorch durante el entrenamiento.

---

### Aplanamiento (Flatten)

Cada imagen original de 28x28 píxeles se transforma en un vector de **784 valores** (`28*28`).  
Este proceso, llamado **aplanamiento**, permite conectar todos los píxeles directamente a la capa de entrada de la red neuronal.

Ejemplo de transformación:
```
Entrada:  (1, 28, 28)
Salida:   (1, 784)
```

De esta forma, cada imagen se convierte en una lista lineal de características lista para ser procesada por el modelo.

---

## Definición del modelo y entrenamiento

El modelo se define con `nn.Sequential()`, una forma sencilla de construir redes en PyTorch agregando capas de manera secuencial.  
En este caso, el flujo del modelo es el siguiente:

1. **Flatten**: aplana la imagen (28x28 → 784).  
2. **Linear (784 → 512)**: primera capa totalmente conectada, que aprende representaciones intermedias.  
3. **Sigmoid**: introduce una activación no lineal para permitir que el modelo aprenda relaciones más complejas.  
4. **Linear (512 → 1)**: capa de salida con una sola neurona, que representa la probabilidad de que la imagen pertenezca a la clase 1.  
5. **Sigmoid final**: convierte el resultado en un valor entre 0 y 1.

La **función de pérdida** utilizada es `nn.BCELoss()` (Binary Cross Entropy), que mide el error entre las predicciones y las etiquetas reales en tareas binarias.  
El **optimizador** es `torch.optim.SGD`, que ajusta los pesos del modelo mediante **descenso del gradiente** con una tasa de aprendizaje definida (`lr=0.001`).

Durante el **entrenamiento**, el modelo aprende repitiendo un ciclo de pasos:

- Calcula las predicciones (`forward pass`).
- Evalúa el error con la función de pérdida.
- Calcula los gradientes (`backward()`).
- Actualiza los pesos (`step()`).
- Repite el proceso por un número determinado de iteraciones o épocas.

Este procedimiento permite que el modelo minimice la pérdida y mejore su precisión en las predicciones.

---

## Evaluación y pruebas individuales

Una vez entrenado, el modelo se pone en modo evaluación con `modelo.eval()`.  
Durante esta fase, se desactiva el cálculo de gradientes (`torch.no_grad()`) para hacer el proceso más rápido y eficiente.

El conjunto de prueba se pasa por el modelo, que genera una **probabilidad** para cada imagen.  
Luego se aplica un **umbral de 0.5**:  
- Si la probabilidad es mayor a 0.5 → se predice clase **1**  
- Si es menor o igual a 0.5 → se predice clase **0**

Comparando las predicciones con las etiquetas verdaderas, se calcula la **precisión** del modelo, es decir, el porcentaje de aciertos sobre el total de imágenes evaluadas.

También se incluye una **prueba individual**, en la cual se selecciona una imagen de ejemplo del conjunto de prueba.  
La imagen se muestra en pantalla junto con su etiqueta real, y el modelo predice si pertenece a la clase 0 o 1.  
Esto permite observar de manera visual cómo el modelo clasifica ejemplos reales.

---

## Funciones y módulos usados

- **`transforms.ToTensor()`** → convierte las imágenes en tensores y normaliza los valores a 0–1.  
- **`datasets.MNIST()`** → carga o descarga el conjunto de datos MNIST.  
- **`.float()`** → cambia el tipo de dato a flotante, necesario para operaciones con PyTorch.  
- **`.reshape()` / `.view()`** → cambia la forma del tensor (por ejemplo, de 28x28 a 784).  
- **`nn.Sequential()`** → construye un modelo capa por capa.  
- **`nn.Linear()`** → capa lineal que realiza una combinación lineal de las entradas.  
- **`nn.Sigmoid()`** → función de activación que mapea valores a probabilidades (0–1).  
- **`nn.BCELoss()`** → calcula la pérdida para clasificación binaria.  
- **`torch.optim.SGD()`** → optimizador de descenso por gradiente estocástico.  
- **`.zero_grad()`** → reinicia los gradientes antes de un nuevo paso de entrenamiento.  
- **`.backward()`** → calcula los gradientes de la pérdida respecto a los pesos.  
- **`.step()`** → actualiza los parámetros del modelo según los gradientes calculados.  
- **`modelo.eval()`** → cambia el modelo a modo evaluación.  
- **`torch.no_grad()`** → desactiva el cálculo de gradientes durante la inferencia.  
- **`plt.imshow()`** → muestra imágenes en escala de grises.  
- **`plt.title()`** y **`plt.show()`** → añaden título y visualizan la figura.

---

## Conclusión

Este notebook demuestra el ciclo completo de una red neuronal simple:  
**cargar, preparar, entrenar, evaluar y probar** un modelo de aprendizaje supervisado.  
A través del ejemplo de MNIST, se ilustran los conceptos clave del aprendizaje de máquina:  
**tensores, funciones de activación, pérdida, optimización y evaluación**, todo dentro de un flujo práctico y fácil de entender.
