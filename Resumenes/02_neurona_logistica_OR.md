# 🧠 ¿Cómo se comporta una neurona logística? (Explicación para compañeros)

Este archivo explica paso a paso cómo funciona y por qué se comporta así una **neurona logística** en PyTorch, usando como ejemplo una compuerta lógica **OR**. Está pensado para que puedas **contarlo como una conversación con un compañero**, con intuición y ejemplos.

---

## 🎬 1. ¿Qué es esta neurona y qué intenta hacer?

> Lo que estamos modelando aquí es **una neurona logística**, que básicamente intenta **aprender la función lógica OR**.  
> Es decir, le damos un par de entradas (por ejemplo: 0 y 1) y tiene que responder si eso representa un 1 (verdadero) o un 0 (falso),  
> igualito que una compuerta lógica OR.

---

## ⚙️ 2. ¿Qué hace la neurona por dentro?

> Por dentro, la neurona toma las dos entradas, las multiplica por dos pesos \( w_1, w_2 \), les suma un sesgo \( b \),  
> y luego le aplica una función llamada **sigmoide** que convierte el resultado en una probabilidad entre 0 y 1.

\[
z = w_1 x_1 + w_2 x_2 + b \\ \hat{y} = \sigma(z)
\]

> La idea es que, si \( z \) es muy grande, la neurona diga ‘esto parece un 1’, y si \( z \) es pequeño o negativo, diga ‘esto parece un 0’.  
> Entonces la **sigmoide** le da ese comportamiento suave entre 0 y 1.

---

## 🧪 3. ¿Cómo aprende?

> La neurona empieza con **pesos aleatorios**, y al principio **no tiene idea de qué es la OR**.  
> Lo que hacemos es mostrarle los 4 casos posibles: (0,0), (0,1), (1,0), (1,1), y sus salidas deseadas: 0, 1, 1, 1.

> Después de cada intento, medimos **qué tanto se equivocó** usando una función llamada **entropía cruzada** (BCELoss),  
> y calculamos **cuánto cambiarían los pesos** para que la próxima vez se equivoque menos.

---

## 🔁 4. ¿Por qué repite tantas veces?

> Porque este proceso es **iterativo**.  
> No aprende todo de un jalón, sino que va ajustando los pesos poquito a poquito en cada vuelta,  
> siempre en la dirección que **disminuye el error**.  
> Eso es lo que hace el **gradiente descendente**:  
> va bajando la montaña de error hasta encontrar el punto más bajo (mínima pérdida).

---

## 🧮 5. ¿Qué papel tiene cada función del código?

> Por ejemplo:
> - `nn.Linear(2,1)` define la parte **lineal** de la neurona: calcula el \( z \).
> - `nn.Sigmoid()` es la parte **no lineal**: convierte eso en una probabilidad.
> - `nn.BCELoss()` compara lo que predice la neurona con el valor real.
> - `optimizer.step()` es quien **mueve los pesos** para mejorar la predicción.

> Y `loss.backward()` es como decirle a PyTorch:  
> “Dime hacia dónde mover los pesos para mejorar”.  
> Es ahí donde PyTorch **calcula automáticamente las derivadas**.

---

## 📈 6. ¿Qué patrón sigue todo esto?

> En realidad, sigue el patrón clásico del aprendizaje supervisado:  
> tienes datos de entrada y sabes qué salida deberían producir,  
> y el modelo va ajustando sus parámetros **para acercarse a esas respuestas**.

> Este patrón no es exclusivo de la función OR;  
> se puede aplicar a clasificar correos spam, detectar rostros, diagnosticar enfermedades…  
> Lo que cambia es cuántas entradas tienes, cuántas neuronas, y cómo conectas todo.

---

## ✅ 7. ¿Cómo sé que ya aprendió?

> Al final de todo, le pasamos otra vez los mismos datos (0,0), (0,1), etc.,  
> y vemos que ahora sí predice valores cercanos a 0 o a 1 según lo esperado.

> También podemos imprimir los **pesos finales**, y si ves que ambos pesos son positivos y el sesgo es negativo,  
> es buena señal: eso significa que la neurona aprendió que basta que **uno solo de los dos inputs** sea 1 para activar la salida.

---

## 🔍 Ejemplo visual (intuición)

> Imagínate que la neurona traza una recta que divide el plano en dos zonas:  
> de un lado, todo lo que considera ‘cero’, y del otro, todo lo que considera ‘uno’.  
> Su trabajo es **mover esa línea** (cambiando los pesos y el sesgo)  
> hasta que separe correctamente los puntos de clase 0 y clase 1.  
> En la función OR, esa separación es sencilla, por eso una sola neurona puede hacerlo.

---

## 🧠 ¿Cómo lo explico en una frase?

> "Una neurona logística **multiplica y suma las entradas**,  
> luego pasa ese valor por una **sigmoide** para obtener una probabilidad,  
> y con **gradiente descendente** va ajustando sus pesos para que sus salidas se parezcan cada vez más a las verdaderas.  
> Lo repite muchas veces, y así aprende a clasificar."

---