# 🧩 Convolución 2D en imágenes

Este notebook demuestra **cómo funciona la operación de convolución** en imágenes usando **PyTorch** y **torchvision** con dos ejemplos:  
1) una **matriz 6×6** sencilla para entender la mecánica, y  
2) una **imagen FashionMNIST (28×28)** donde aplicamos filtros clásicos (Sobel/Prewitt).

---

## 🎯 Objetivo
- Comprender **qué es la convolución**, cómo se **aplica un kernel** y qué efecto produce (bordes, suavizado, realce).
- Relacionar la operación con lo que hacen las **CNN** para **extraer características**.

---

## 🧠 ¿Qué es la convolución?
La **convolución 2D** combina una imagen $(I)$ con un **kernel** $(K)$ (matriz pequeña como 3×3).  
Para cada posición del kernel sobre la imagen:
1. Se alinean sus celdas con los píxeles vecinos.
2. Se hace **multiplicación elemento a elemento**.
3. Se **suman** los productos y se escribe el resultado en la salida.

Formalmente, para una posición $(i, j)$:
$$
(I * K)[i,j] = \sum_{u=-a}^{a} \sum_{v=-b}^{b} I[i+u, j+v] \cdot K[u,v]

$$
donde $(a,b)$ dependen del tamaño del kernel (por ejemplo, $(a=b=1)$ para 3×3).

**Padding (relleno)** controla el tamaño de la salida:  
- `padding=0` → salida más pequeña (p.ej. 6→4).  
- `padding=1` con kernel 3×3 → **conserva tamaño** (28→28).

> En CNN, varios kernels se aprenden durante el entrenamiento para detectar **bordes, texturas y patrones** en distintas orientaciones/escala.

---

## 🔁 Flujo del proceso (resumen)
1. **Preparación de datos**  
   - Convertimos la imagen a **tensor** y (opcionalmente) a **escala de grises** (1 canal).
   - Ajustamos dimensiones a **(N, C, H, W)** para `Conv2d`.
2. **Definición de kernels**  
   - Especificamos matrices 3×3 para **bordes horizontales/verticales/diagonales**.
3. **Aplicación de la convolución**  
   - Usamos `nn.Conv2d` con nuestros **pesos cargados manualmente**.
4. **Visualización**  
   - Mostramos **imagen original** y **resultado** para interpretar efectos.

---
## 🧪 Ejemplos incluidos

### 🧮 1) Matriz 6×6 (toy example)

-   Imagen: franja vertical de intensidad alta (`10`) y el resto `0`.\
-   Kernel: `[[1, 0, -1], [1, 0, -1], [1, 0, -1]]` (bordes verticales).\
-   Sin padding (`padding=0`) → salida de **4×4**.

🔍 **Qué se observa:** - Los valores grandes aparecen **donde hay cambio
brusco** (de 10 a 0).\
- Las regiones uniformes (todo 10 o todo 0) generan salida ≈ 0.\
- Es un ejemplo ideal para **ver cómo se "mueve" el kernel** y por qué
la salida cambia de tamaño.

🧩 **Interpretación:**\
La convolución actúa como un detector de transición: el signo (+/--) del
resultado indica **la dirección del contraste** (claro→oscuro o
viceversa).

------------------------------------------------------------------------

### 👕 2) FashionMNIST (28×28)

-   Dataset: conjunto de prendas en escala de grises (0--255 →
    normalizado 0--1).\
-   Imagen elegida (índice 4) → categoría "Abrigo".\
-   Se aplican tres filtros:
    -   **Sobel Horizontal:** detecta bordes arriba/abajo.\
    -   **Sobel Vertical:** detecta bordes izquierda/derecha.\
    -   **Filtro Diagonal (Prewitt):** resalta diagonales y bordes
        oblicuos.\
-   `padding=1` → la salida mantiene 28×28.

🔍 **Qué se observa:** - Sobel Horizontal destaca los **contornos
horizontales del abrigo** (hombros y parte inferior).\
- Sobel Vertical resalta los **bordes laterales** de la prenda.\
- El filtro Diagonal muestra líneas suaves en ángulo, **combinando
direcciones**.

📊 **Análisis:** - Cada filtro capta una **orientación distinta** de los
bordes.\
- En redes convolucionales, estos mapas (feature maps) se combinan y
profundizan capa a capa para construir una **representación jerárquica**
de la imagen (de bordes → texturas → formas → objetos).
---


## 🛠️ Funciones y clases usadas (qué hacen)

### 🔹 Transformaciones y datasets

-   `transforms.Compose([...])`: encadena transformaciones.\
-   `transforms.ToTensor()`: convierte a tensor y normaliza a `[0, 1]`.\
-   `transforms.Grayscale()`: convierte la imagen a 1 canal (escala de
    grises).\
-   `torchvision.datasets.FashionMNIST(root, train, download, transform)`:
    descarga/carga el dataset.

### 🔹 Tensores y convolución

-   `tensor.view(N, C, H, W)`: reinterpreta dimensiones sin copiar
    datos.\
-   `tensor.unsqueeze(dim)`: añade una nueva dimensión (por ejemplo, el
    batch).\
-   `nn.Conv2d(in_ch, out_ch, kernel_size, padding, bias)`: crea la capa
    de convolución 2D.\
-   `layer.weight`: pesos del kernel (forma `(out_ch, in_ch, kH, kW)`).\
-   `tensor.squeeze()`: elimina dimensiones de tamaño 1.\
-   `tensor.detach()`: desconecta del grafo de gradientes (para
    visualización).

### 🔹 Visualización

-   `plt.figure(figsize=(w, h))`: define tamaño de la figura.\
-   `plt.subplot(r, c, i)`: organiza imágenes en cuadrícula.\
-   `plt.imshow(img, cmap='gray')`: muestra imagen en escala de grises.\
-   `plt.axis('off')`: oculta ejes.\
-   `plt.tight_layout()`: ajusta espacios entre subplots.
---

## 🧩 Interpretación de resultados
- **Sobel Horizontal**: resalta cambios **arriba/abajo** → bordes horizontales.  
- **Sobel Vertical**: resalta cambios **izquierda/derecha** → bordes verticales.  
- **Diagonal/Prewitt**: marca **diagonales** y estructuras oblicuas.  
- Las intensidades pueden ser **positivas/negativas** según la dirección del contraste; al visualizar, es común **re-escalar** o tomar magnitudes.


---

## ✅ Conclusión

La **convolución 2D** es el corazón del procesamiento visual moderno.\
Con un simple recorrido local y suma ponderada, se pueden **extraer
patrones estructurales** que luego alimentan modelos más complejos.\
Estos conceptos son exactamente los que las **redes neuronales
convolucionales (CNN)** usan para aprender automáticamente **qué
patrones son relevantes** según los datos.

---