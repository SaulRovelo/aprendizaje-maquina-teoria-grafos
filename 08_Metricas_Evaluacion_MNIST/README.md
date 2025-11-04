# 📊 Métricas de Evaluación del Modelo (MNIST 0 y 1)

Este notebook amplía el proyecto anterior de la **red neuronal para reconocer los dígitos 0 y 1** del conjunto MNIST, incorporando ahora un paso esencial del flujo de aprendizaje automático: **la evaluación del modelo mediante métricas**.

---

## 🧠 ¿Qué son las métricas de evaluación?

Las **métricas de evaluación** son indicadores que permiten medir el **desempeño real de un modelo entrenado**.  
No basta con entrenar y obtener una pérdida baja: también es necesario **verificar qué tan bien el modelo predice en los datos de prueba** y cómo se comporta en cada clase.

En esta práctica se utilizan tres métricas fundamentales:

### 🔹 Accuracy (Precisión global)
Indica el **porcentaje total de aciertos** del modelo.  
Se calcula comparando todas las predicciones con las etiquetas reales:

$$[
Accuracy = \frac{\text{Predicciones correctas}}{\text{Total de muestras}} \times 100
]
$$

Es una medida general, pero no distingue entre clases desbalanceadas.

---

### 🔹 Reporte de Clasificación
Generado con `classification_report()` de *scikit-learn*, muestra tres métricas por clase:

- **Precision:** porcentaje de aciertos sobre todas las predicciones positivas.  
- **Recall:** porcentaje de verdaderos positivos detectados correctamente.  
- **F1-score:** promedio armónico entre precisión y recall, equilibra ambos valores.

Esta herramienta permite identificar **si el modelo favorece una clase sobre otra**.

---

### 🔹 Matriz de Confusión
Generada con `confusion_matrix()`, muestra los **aciertos y errores por clase** en forma de tabla:

|               | Predicción 0 | Predicción 1 |
|---------------|--------------|--------------|
| **Real 0**    | Verdaderos negativos | Falsos positivos |
| **Real 1**    | Falsos negativos | Verdaderos positivos |

Se visualiza con un *heatmap* mediante **Seaborn**, facilitando la interpretación de los resultados.

---

## 🧩 Conclusión

Las métricas de evaluación son esenciales para **interpretar el rendimiento real del modelo**, detectar sesgos y mejorar futuras versiones.  
Gracias a estas herramientas, podemos ir más allá del simple “acierta o falla” y entender **cómo y por qué el modelo toma sus decisiones**.

---
