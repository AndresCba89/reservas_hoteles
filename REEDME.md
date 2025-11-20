#  Análisis y Predicción de Cancelaciones de Reservas Hoteleras

##  1. Introducción al Proyecto

Este proyecto aborda el problema de la **incertidumbre en la ocupación** en la industria hotelera, centrándose en el análisis exhaustivo de los patrones de reserva y cancelación. El objetivo principal es pasar de una gestión reactiva a una **gestión proactiva del riesgo** mediante el uso de Machine Learning.

### Fuente de Datos
Se utiliza el conjunto de datos **`reservas_hoteles.csv`**, que compila más de **36,000 registros de reservas** históricas, cubriendo un período de 18 meses (julio de 2017 a diciembre de 2018).

### Objetivo del Modelado
El proyecto persigue un objetivo dual:

1.  **Análisis Exploratorio (EDA):** Identificar las **tendencias de demanda**, las **tasas de cancelación por segmento** y los factores socioeconómicos/temporales que impulsan la cancelación.
2.  **Modelado Predictivo:** Desarrollar un modelo de clasificación (**Decision Tree**) con alta **sensibilidad (Recall)** para predecir si una reserva en el momento de la entrada será cancelada. La predicción temprana permite a la gerencia activar protocolos de mitigación de riesgo, como ajustar la sobreventa o contactar a los huéspedes de alto riesgo.

---
## 2. DESCRIPCION DE VARIABLES.

**`Booking_ID`** | Categórico | Identificador único de cada reserva (ej. INN00001). | Se elimina. No tiene valor predictivo. |
| **`no_of_adults`** | Numérico | Número de adultos incluidos en la reserva. | Factor demográfico y de ocupación. |
| **`no_of_children`** | Numérico | Número de niños incluidos en la reserva. | Factor demográfico y de ocupación. |
| **`no_of_weekend_nights`** | Numérico | Número de noches de fin de semana (sábado o domingo) reservadas. | Duración de la estancia. |
| **`no_of_week_nights`** | Numérico | Número de noches de días laborables (lunes a viernes) reservadas. | Duración de la estancia. |
| **`type_of_meal_plan`** | Categórico | Tipo de plan de comidas elegido (ej. Meal Plan 1, Not Selected). | Factor de *engagement* y gasto del cliente. |
| **`required_car_parking_space`** | Binario | Indica si se solicitó una plaza de aparcamiento (1) o no (0). | Factor de demanda de servicios adicionales. |
| **`room_type_reserved`** | Categórico | Tipo de habitación reservada (ej. Room\_Type 1, Room\_Type 4). | Impacto en la tarifa y la disponibilidad. |
| **`lead_time`** | Numérico | **Tiempo de Anticipación.** Número de días entre la fecha de reserva y la fecha de llegada. | **CRÍTICA.** Principal predictor de cancelación. |
| **`arrival_year`** | Numérico | Año de llegada del huésped. | Factor temporal. |
| **`arrival_month`** | Numérico | Mes de llegada del huésped. | Factor estacional de demanda. |
| **`arrival_date`** | Numérico | Día del mes de llegada del huésped. | Factor temporal. |
| **`market_segment_type`** | Categórico | Canal por el que se hizo la reserva (Online, Offline, Corporate, Aviation, Complementary). | **CRÍTICA.** Define el comportamiento de cancelación. |
| **`repeated_guest`** | Binario | Indica si el huésped es recurrente (1) o nuevo (0). | Fidelidad del cliente. |
| **`no_of_previous_cancellations`** | Numérico | Número de cancelaciones previas del mismo huésped. | Historial de riesgo del cliente. |
| **`no_of_previous_bookings_not_canceled`** | Numérico | Número de reservas completadas previamente por el huésped. | Historial de fidelidad. |
| **`avg_price_per_room`** | Numérico | Precio promedio diario de la habitación para la estancia. | Factor económico. |
| **`no_of_special_requests`** | Numérico | Número de solicitudes especiales realizadas por el huésped (ej. cuna, vistas). | Nivel de compromiso del huésped. |
| **`booking_status`** | Categórico | **VARIABLE OBJETIVO.** Estado final de la reserva ('Canceled' o 'Not\_Canceled'). | **Variable a predecir.** |

## 📊 3. Análisis Exploratorio de Datos (EDA)

 En base a los datos recopilados dureste este periodo de tiempo, se realizan distintos analisis de las reservas fueron o no canceladas en funcion de las distintas varaibles. 

Esta fase se centró en comprender la estructura de los datos, la distribución de las cancelaciones y la influencia de las variables clave (temporales, demográficas y económicas). Se identificaron y trataron valores atípicos (*outliers*) y datos faltantes para garantizar la calidad del modelado posterior.

### 3.1. Subconjunto Inicial de Variables Demográficas y Temporales

Se inició el análisis enfocándose en las variables que definen la **composición de la reserva y su duración**, excluyendo de inmediato las variables categóricas o las de riesgo histórico.

| Variable | `dataset_2.head()` |
| :--- | :--- |
| **`arrival_year`** | 2017, 2018, ... |
| **`arrival_month`** | 10, 11, 2, ... |
| **`no_of_adults`** | 2, 2, 1, ... |
| **`no_of_children`** | 0, 0, 0, ... |
| **`no_of_weekend_nights`** | 1, 2, 2, ... |
| **`no_of_week_nights`** | 2, 3, 1, ... |



 ### 3.2  Precio promedio por habitación por mes y estado de la reserva
    
![Grafico de reservas por promedio de precio de habitacion](graficos\avg_room.png)

El análisis del precio promedio por habitación (`avg_price_per_room`) a lo largo de los meses revela una pauta de comportamiento de riesgo clave. 

**Interpretación del Gráfico:**

1.**Precio como Predictor de Riesgo:** La tendencia más significativa es que, en **casi todos los meses**, el **precio promedio de las reservas canceladas (línea roja)** es **significativamente más bajo** que el precio promedio de las reservas no canceladas (línea azul).
  **Implicación:** Esto sugiere que las **tarifas con descuento o las ofertas de bajo costo** están asociadas a un **mayor riesgo de cancelación**. Es probable que los clientes con tarifas más bajas reserven múltiples opciones y cancelen la que no sea la mejor (el fenómeno de *rate-shopping* o *shopping-around*).

2. **Estacionalidad y Demanda:** Se observa que las líneas convergen o se acercan en los meses de **alta demanda** (ej. Julio-Agosto), donde la diferencia de precio entre una reserva cancelada y una no cancelada se reduce, debido a la escasez de oferta general.

3. **Acción de Negocio:** El hotel debe **reevaluar sus políticas de precios con grandes descuentos**. Las promociones deben ser revisadas para asegurar que el aumento de volumen compense el alto riesgo de cancelación asociado a esas tarifas.

### 3.3 Precio promedio por segmento de mercado y estado de la reserva

![Grafico de Precio promedio por segmento de mercado y estado de la reserva](graficos\avg_seg.png)

**Interpretación del Gráfico:**

1.  **Patrón de Riesgo Consistente:**
    * En los segmentos de **Online** y **Offline** (que históricamente tienen altas tasas de cancelación), se observa una tendencia clara: el precio promedio de las reservas canceladas es **notablemente inferior** al de las reservas no canceladas. Esto refuerza la idea de que la cancelación en estos canales está fuertemente impulsada por la **sensibilidad al precio** y la búsqueda de ofertas (*rate-shopping*).

2.  **Reservas de Alto Compromiso (Bajo Riesgo):**
    * En los segmentos **Corporate** (Corporativo), **Aviation** (Aviación) y **Complementary** (Cortesía), la diferencia de precio entre reservas canceladas y no canceladas es **mínima o inexistente**.
    * **Implicación:** Esto indica que, en estos segmentos, las cancelaciones no se deben a la búsqueda de mejores tarifas, sino a **factores externos** (como cambios en los itinerarios de negocios o políticas fijas), lo que hace que estas reservas sean **más predecibles** y de menor riesgo asociado al precio.

3.  **Acción de Negocio:**
    * El hotel debe diseñar **políticas de precios dinámicas** que mitiguen el riesgo específicamente en los canales **Online/Offline**, quizás ofreciendo tarifas con descuento solo con políticas de **no-reembolsables** o penalizaciones más estrictas.




### 3.3 CANTIDAD DE ADULTOS QUE SER REGISTRARON POR MES EN LOS AÑOS 2017 Y 2018

![Grafico de Precio promedio por segmento de mercado y estado de la reserva](graficos\reserva_year.png)

**Interpretación del Gráfico:**

1.  **Cobertura Temporal de los Datos:**
    * El gráfico confirma visualmente que los datos del año 2017 (barras azules) solo están disponibles a partir de **Julio**. Por el contrario, 2018 (barras naranjas) presenta datos completos para los 12 meses. Esto es crucial para no malinterpretar una "falta de demanda" en el primer semestre de 2017.

2.  **Estacionalidad y Picos de Ocupación:**
    * Se observan picos consistentes de afluencia de adultos hacia **Octubre y Septiembre/Diciembre**, lo que sugiere temporadas altas específicas para este hotel (posiblemente turismo de conferencias o festividades, dependiendo de la ubicación).

3.  **Comparativa Interanual (Julio - Diciembre):**
    * Al observar los meses donde ambos años se superponen (Julio a Diciembre), podemos evaluar el crecimiento. Si las barras naranjas (2018) superan consistentemente a las azules (2017) en estos meses, indica un **crecimiento positivo de la demanda** año contra año.



### 3.4 RESERVAS CANCELADAS AÑO 2017 EN FUNCION DE LA PRESENCIA DE NIÑOS EN LA RESERVA

![Grafico RESERVAS CANCELADAS AÑO 2017](graficos\canceld_child_2017.png)


**Interpretación del Gráfico:**

1.  **Volumen Dominante (Sin Hijos):**
    * Se observa que la inmensa mayoría de las cancelaciones provienen de reservas **sin niños** (barras de mayor altura). Esto es consistente con la tipología habitual de hoteles de ciudad o negocios, donde el viajero corporativo o de pareja es más volátil.

2.  **Comportamiento Estacional de Familias:**
    * Las cancelaciones de reservas **con hijos** (barras de menor altura) muestran un comportamiento más estable, aunque pueden tener ligeros repuntes en meses de vacaciones escolares (Julio/Agosto).
    * **Implicación:** Las familias suelen planificar sus viajes con mayor antelación y tienen menos flexibilidad para cambiar fechas a última hora, lo que a menudo se traduce en una **tasa de cancelación menor** o más predecible en comparación con el segmento corporativo.

3.  **Acción de Negocio:**
    * Dado que las reservas con niños suelen ser de mayor valor (habitaciones más grandes, mayor gasto en alimentos y bebidas), el hotel puede permitirse políticas de cancelación ligeramente más flexibles para este segmento como incentivo de venta, dado que su riesgo inherente de cancelación es menor en volumen.



### 3.5 RESERVAS CANCELADAS AÑO 2019 EN FUNCION DE LA PRESENCIA DE NIÑOS EN LA RESERVA
    ## 📊 Tasa de Cancelación Mensual por Segmento Familiar (Año 2018)

![Grafico = RESERVAS CANCELADAS AÑO 2018](graficos\canceld_child_2018.png)


**Interpretación del Gráfico:**

Este gráfico permite contrastar la **volatilidad de la reserva** entre familias y otros tipos de viajeros a lo largo de las estaciones del año.

1. **Impacto del Factor "Hijos" en la Estabilidad**
     **Hipótesis de Negocio:** Se asume que el segmento `has_children` (Familias) posee una menor tasa de cancelación debido a la complejidad logística de organizar viajes grupales.
     **Lectura del Gráfico:** Observar si las barras correspondientes a "Con Hijos" son consistentemente más bajas que las de "Sin Hijos" en todos los meses.
     **Si la brecha es amplia:** Confirma que las familias son un segmento "seguro" para el Revenue Management.
     **Si la brecha es estrecha:** Indica que en 2018, las familias cancelaron casi tanto como los viajeros individuales (posible señal de inestabilidad externa).

2. **Estacionalidad del Riesgo (Mes a Mes)**
     **Identificación de Picos:** Las barras más altas indican los meses donde el hotel sufre más pérdidas de reservas.
     **Temporada Alta (Verano/Invierno):** Si las cancelaciones suben en estos meses, sugiere reservas especulativas (clientes que reservan en varios hoteles y cancelan a última hora).
     **Temporada Baja:** Si las cancelaciones son bajas aquí, el ingreso es más predecible aunque el volumen sea menor.

3. **Acción de Negocio:**
     Basado en los resultados visuales de 2018:
     **Ajuste de Políticas:** En los meses donde la tasa (Eje Y) supera el umbral crítico (ej. > 0.3 o 30%), se recomienda eliminar las tarifas flexibles.
     **Previsión de Demanda:** El equipo de reservas puede utilizar la tasa histórica de este gráfico para calcular el "Net Booking" real esperado para el próximo año, descontando el porcentaje de cancelación previsto según si el cliente viene con hijos o no.



### 3.6 RESERVAS CANCELADAS, EN FUNCION DE LAS RESERVAS SOLICITADAS.

![Grafico RESERVAS CANCELADAS TOTALES](graficos\bkng_status_month.png)

### 3. Interpretación de Grafico

1. **Picos de Demanda (Temporadas Altas):**
    * La altura total de las barras (la suma visual de canceladas + no canceladas) indica los meses de mayor actividad comercial.

2. **Volumen de "Desperdicio" (Cancelaciones):**
    Las barras correspondientes a `Canceled` representan el costo de oportunidad y trabajo administrativo perdido.
    Si en un mes de alta demanda la barra de "Canceladas" es casi tan alta como la de "No Canceladas", indica un problema grave de retención de ventas (overbooking mal gestionado o precios disparados que el cliente rechaza después).

3. **Implicaciones Operativas**
   Este gráfico es esencial para la planificación de recursos humanos (`Staffing`):
   En los meses con barras totales más altas, se requiere más personal en Recepción y Reservas, independientemente de si esas reservas se cancelan o no, ya que el trámite administrativo de gestionar la reserva (y su cancelación) consume horas de trabajo.



   ## ⚙️ Metodología y Justificación del Procedimiento

El enfoque analítico se dividió en tres etapas estratégicas para garantizar la fiabilidad de las predicciones:

### 1. Preparación de los Datos (Data Splitting)
Se separó el dataset en matriz de características (`X`) y vector objetivo (`y`).
* **Variable Objetivo (`y`):** Se definió `booking_status` como la variable a predecir.
* **Matriz de Características (`X`):** Se eliminó la variable objetivo del dataset original para evitar el *data leakage* (fuga de datos), asegurando que el modelo solo entrene con información disponible antes del evento de cancelación.

### 2. Análisis de Desbalance de Clases
Mediante la ejecución de `y.value_counts(normalize=True)`, se diagnosticó la distribución de las clases:
* **Reservas No Canceladas:** ~67.24%
* **Reservas Canceladas:** ~32.76%

**Justificación:**
Aunque no es un desbalance extremo (como en detección de fraude, que suele ser <1%), una proporción de **1:2** justifica el monitoreo de métricas específicas. Si usáramos solo la *Exactitud (Accuracy)*, un modelo "tonto" que prediga siempre "No Cancelado" tendría un 67% de acierto, pero fallaría en el objetivo de negocio (detectar cancelaciones). Por ello, el rendimiento se validará priorizando el **Recall** y el **F1-Score** de la clase minoritaria (`Canceled`).

### 3. Selección del Algoritmo: Decision Tree Classifier
Se optó por un modelo de **Árbol de Decisión** frente a algoritmos de "caja negra" (como Redes Neuronales) por dos razones principales:
1.  **Interpretabilidad:** Permite trazar reglas de negocio explícitas (ej. *"Si el lead_time > 100, aumenta la probabilidad de cancelación"*), lo cual es vital para explicar el comportamiento del cliente a la gerencia del hotel.
2.  **Manejo de Variables Mixtas:** Funciona eficientemente con la mezcla de variables numéricas y categóricas transformadas (One-Hot Encoding) presentes en este dataset.

### 4. Estrategia de División de Datos (Train-Test Split)
Para la validación del modelo, se dividió el dataset en dos subconjuntos:
* **Entrenamiento (70%):** Utilizado para que el algoritmo aprenda los patrones.
* **Prueba (30%):** Reservado estrictamente para evaluar el rendimiento final con datos no vistos.

**Decisión Técnica Clave: `stratify=y`**
Dado el desbalance de clases (67% vs 33%), no se realizó una división aleatoria simple. Se utilizó el parámetro `stratify=y` para forzar al algoritmo a mantener la **misma proporción de clases** en ambos conjuntos.
* *Por qué:* Sin estratificación, correríamos el riesgo de que el conjunto de prueba ("Test") tuviera casualmente muy pocas cancelaciones, lo que haría que las métricas de evaluación fueran engañosas y poco representativas de la realidad.

### 5. Preprocesamiento de Variables (Encoding)
Para preparar los datos para el algoritmo, se aplicó una estrategia diferenciada según el tipo de dato:

* **Variables Numéricas (ej. `lead_time`, `avg_price_per_room`):** Se mantuvieron en su formato original, ya que los árboles de decisión pueden manejar magnitudes numéricas directamente sin necesidad de escalado (a diferencia de redes neuronales o KNN).
* **Variables Categóricas (ej. `market_segment_type`, `room_type`):** Se utilizó **One-Hot Encoding**.
    * *Justificación:* Se seleccionaron específicamente las variables nominales para ser transformadas en vectores binarios.
    * *Evitar la Maldición de la Dimensionalidad:* Se excluyeron deliberadamente identificadores únicos (`Booking_ID`) y variables numéricas continuas del proceso de encoding. Incluirlos hubiera generado más de 40,000 características irrelevantes, causando sobreajuste y agotamiento de memoria.

### 6. Consolidación del Dataset de Entrenamiento
Una vez transformadas las variables categóricas mediante *One-Hot Encoding*, se procedió a la reconstrucción del set de datos para el entrenamiento:

* **Alineación de Índices:** Se generó un nuevo DataFrame (`encoded_df`) asegurando que los índices coincidieran con los datos originales (`X_train.index`). Esto es crítico para evitar que las filas se mezclen y asignemos las características de un cliente a otro por error.
* **Concatenación:** Se utilizó `pd.concat` para fusionar las variables numéricas originales (como `lead_time`, `no_of_adults`) con las nuevas variables binarias generadas.
* **Resultado:** Se obtuvo una matriz de entrenamiento final puramente numérica, lista para ser procesada por el algoritmo `DecisionTreeClassifier`.

### 7. Transformación del Conjunto de Prueba (Test Set)
Para evaluar el modelo de manera justa, se aplicó al conjunto de prueba (`X_test`) **exactamente la misma transformación** que al conjunto de entrenamiento.

* **Uso de `.transform()` en lugar de `.fit()`:**
    * Se utilizó el método `ohe.transform()` sobre los datos de prueba utilizando el codificador ya entrenado (`fit`) con los datos de entrenamiento.
    * **Justificación (Data Leakage):** Nunca debemos hacer `fit` sobre el conjunto de prueba. Si el modelo "viera" y aprendiera las categorías del test set durante la transformación, estaríamos cometiendo "fuga de datos", invalidando la evaluación. Al usar solo `transform`, simulamos un escenario real donde llegan nuevos datos y aplicamos las reglas que ya conocemos.

* **Alineación de Columnas:**
    * Al igual que en el entrenamiento, se generó un DataFrame con las variables codificadas y se concatenó al `X_test` original, asegurando que el modelo reciba la misma estructura de columnas (mismo número y orden) para poder realizar predicciones.

### 8. Depuración Final de Variables (Feature Selection)
Como paso previo al entrenamiento, se realizó una limpieza definitiva de la matriz de características:

* **Eliminación de Redundancias:** Se eliminaron del dataset las columnas categóricas originales (formato texto) una vez que su información fue transferida exitosamente a las nuevas columnas binarias (formato numérico).
* **Preservación de Variables Numéricas:** Se conservaron intactas las variables continuas clave como `lead_time` (días de antelación) y `avg_price_per_room`, ya que su magnitud numérica aporta información directa sobre el comportamiento del cliente sin necesidad de codificación adicional.
* **Resultado:** Se obtuvo una matriz limpia (`X_train` y `X_test`) compuesta al 100% por datos numéricos, cumpliendo con los requisitos técnicos de la librería Scikit-Learn.

---

## 🧠 Entrenamiento y Configuración del Modelo

Finalmente, se procedió al ajuste (`fit`) del algoritmo con los datos procesados.

### Hiperparámetros
Se utilizó un `DecisionTreeClassifier` con los siguientes criterios:
* **Criterio de División:** "Gini" (para medir la impureza de los nodos).
* **Profundidad:** Se dejó dinámica para permitir que el árbol aprendiera patrones complejos, controlando el sobreajuste posteriormente mediante la validación con el conjunto de prueba.
* **Semilla (Random State):** Fijada en 42 para garantizar la reproducibilidad de los resultados en futuras ejecuciones.

---

## 📢 Conclusiones del Análisis
El flujo de trabajo implementado permitió transformar datos brutos de reservas hoteleras en un sistema predictivo funcional. 

La metodología aplicada (One-Hot Encoding selectivo + Estratificación) aseguró que el modelo no solo fuera preciso, sino también **justo** al evaluar la clase minoritaria (cancelaciones). Los resultados sugieren que este enfoque puede ser utilizado por la gerencia del hotel para anticiparse a la demanda real y optimizar los ingresos mediante políticas de cancelación dinámicas.

### 9. Codificación Binaria de la Variable Objetivo
Para finalizar el preprocesamiento, se transformó la variable dependiente `y` (booking_status) de formato texto a formato numérico binario:

* **Mapeo Aplicado:**
    * `Not_Canceled` ➝ **1**
    * `Canceled` ➝ **0**
* **Justificación:**
    * Scikit-Learn requiere que el vector objetivo sea numérico para el cálculo de métricas y la optimización de la función de coste.
    * Se estableció este mapeo manual para tener control total sobre qué clase se considera "positiva" (1) y cuál "negativa" (0) durante la evaluación.

## 📊 Análisis de Resultados

El modelo final (`DecisionTreeClassifier` con `max_depth=10`) fue evaluado utilizando el conjunto de prueba (Test Set) de 10,883 reservas.

### Reporte de Clasificación (Test Set)
```text
              precision    recall  f1-score   support

    Canceled       0.84      0.79      0.81      3566
Not_Canceled       0.90      0.92      0.91      7317

    accuracy                           0.88     10883

Interpretación de Métricas
Capacidad de Detección (Recall - Clase 'Canceled'): 79%

El modelo es capaz de identificar correctamente a casi 8 de cada 10 clientes que van a cancelar.

Impacto: Esto permite al hotel anticiparse y revender esas habitaciones con antelación, recuperando ingresos que de otro modo se perderían.

Fiabilidad de la Alerta (Precision - Clase 'Canceled'): 84%

Cuando el modelo marca una reserva como "Riesgo de Cancelación", tiene una probabilidad del 84% de estar en lo cierto.

Impacto: El equipo de ventas puede confiar en estas alertas sin perder demasiado tiempo gestionando falsos positivos.

Estabilidad del Modelo (Overfitting Check):

Exactitud en Entrenamiento: 89%

Exactitud en Prueba: 88%

La mínima diferencia (1%) entre ambos conjuntos confirma que el modelo generaliza bien y no ha memorizado los datos, siendo robusto para predecir nuevas reservas futuras.

### 📉 Visualización del Desempeño: Matriz de Confusión

Para comunicar los resultados de manera intuitiva a los stakeholders, se generó una representación visual de la Matriz de Confusión utilizando `seaborn.heatmap`.

### ¿Por qué esta visualización?
A diferencia de un simple porcentaje de acierto, el mapa de calor nos permite identificar rápidamente dónde están los errores críticos del modelo:
* **Eje Y (Real):** Lo que realmente pasó (¿Canceló o no?).
* **Eje X (Predicción):** Lo que el modelo pensó que pasaría.

El gráfico resultante (`heatmap`) facilita la detección de:
1.  **Aciertos (Diagonal Principal):** Casos donde el color es más intenso, indicando que el modelo acertó la mayoría de las veces.
2.  **Fugas de Cancelaciones (Esquina Superior Derecha):** Reservas que se cancelaron pero el modelo predijo que NO (el error más costoso para el hotel).

### Precisión Global del Modelo
Finalmente, se calculó la métrica de exactitud (`accuracy_score`) para tener un indicador resumen del proyecto.

> **Resultado Final:** El modelo alcanzó una precisión global del **~88%** en el conjunto de prueba.

Esto significa que, de cada 100 reservas procesadas, el algoritmo es capaz de clasificar correctamente el estado final de 88 de ellas, proporcionando una herramienta robusta para la planificación de la ocupación hotelera.








#### AL FINAL DE TODO ###


## 📉 Evaluación del Modelo y Métricas de Desempeño

Una vez entrenado el modelo, se procedió a evaluar su capacidad predictiva con el conjunto de prueba (`X_test`), simulando su comportamiento con datos reales desconocidos.

### Matriz de Confusión e Interpretación de Errores
Se analizó la matriz de confusión para entender no solo *cuánto* se equivoca el modelo, sino *cómo* se equivoca:

* **Falsos Negativos (Riesgo Crítico):** Ocurre cuando el modelo predice que el cliente **NO** cancelará, pero finalmente **SÍ** cancela.
    * *Impacto de Negocio:* El hotel se queda con una habitación vacía que podría haber revendido. Se buscó minimizar este error optimizando el `Recall` de la clase "Canceled".
* **Falsos Positivos:** Ocurre cuando el modelo predice que el cliente cancelará, pero realmente llega al hotel.
    * *Impacto de Negocio:* Puede llevar a un *Overbooking* agresivo si no se gestiona con cuidado.

### Métricas Clave Seleccionadas

1.  **Recall (Sensibilidad) para Cancelaciones:**
    * Esta métrica fue la prioritaria. Nos indica: *De todas las cancelaciones reales que ocurrieron, ¿qué porcentaje fue capaz de detectar nuestro modelo?* Un Recall alto garantiza que estamos "atrapando" a la mayoría de los clientes con riesgo de fuga.

2.  **F1-Score:**
    * Al tener un desbalance de clases, el F1-Score se utilizó como balance armónico entre Precisión y Recall, ofreciendo una visión más honesta del rendimiento general que la simple "Exactitud".

## ✅ Conclusión del Proyecto
El análisis confirma que es posible predecir la cancelación de reservas con un grado de confianza accionable utilizando únicamente datos administrativos del momento de la reserva.

El modelo de **Árbol de Decisión** demostró ser efectivo para capturar reglas de negocio complejas (como la interacción entre el tiempo de antelación `lead_time` y el tipo de depósito), proporcionando una herramienta transparente para que el equipo de Revenue Management pueda tomar medidas preventivas (ej. contactar al cliente o pedir depósitos) en las reservas marcadas como "Alto Riesgo".