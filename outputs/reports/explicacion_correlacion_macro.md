Análisis de Correlación: Recaudo vs Variables Macroeconómicas
**Tesis de Maestría — Rentas Cedidas | Periodo: Oct 2021 – Dic 2025**

---

## 1. Descripción de la Gráfica

La **matriz de correlación** muestra los coeficientes de Pearson (r) entre 7 variables del sistema de recaudo:

| Variable | Descripción |
|---|---|
| **Recaudo_Neto** | Recaudo mensual neto de rentas cedidas ($COP) |
| **lag_12** | Recaudo del mismo mes del año anterior |
| **IPC** | Variación anual del Índice de Precios al Consumidor (%) |
| **Salario_Minimo** | Variación anual del salario mínimo legal (%) |
| **UPC** | Variación anual de la Unidad de Pago por Capitación (%) |
| **YoY_Recaudo** | Crecimiento interanual del recaudo (%) |
| **IEP** | Índice de Eficiencia del Portafolio = YoY_Recaudo / UPC |

Se muestra solo el **triángulo inferior** (la diagonal superior es simétrica). La escala de color RdBu va de **r = −1** (azul, inversamente proporcional) a **r = +1** (rojo oscuro, directamente proporcional).

---

## 2. Hallazgos Clave

### 🔴 Correlaciones Fuertes (|r| > 0.70)

| Par de Variables | r | Interpretación |
|---|---|---|
| **Recaudo_Neto ↔ lag_12** | **0.875** | El recaudo actual está altamente determinado por el recaudo del mismo mes del año anterior. Esto confirma un **patrón estacional anual repetitivo** |
| **Salario_Minimo ↔ UPC** | **0.903** | Las dos variables de costos están altamente correlacionadas entre sí. Se mueven juntas porque ambas se ajustan anualmente con base en indicadores similares (inflación, productividad) |
| **YoY_Recaudo ↔ IEP** | **0.927** | Correlación esperada: el IEP es una transformación directa del crecimiento interanual. No es una relación independiente |

### 🟡 Correlaciones Moderadas (0.30 < |r| < 0.70)

| Par de Variables | r | Interpretación |
|---|---|---|
| **IPC ↔ Salario_Minimo** | **0.614** | La inflación influye moderadamente en los incrementos salariales. Relación económica esperada |
| **Recaudo_Neto ↔ YoY_Recaudo** | **0.430** | Cuando el nivel de recaudo es alto, el crecimiento interanual tiende a ser mayor. Sugiere un **efecto compuesto** de crecimiento |
| **Recaudo_Neto ↔ IEP** | **0.370** | El recaudo tiende a ser más eficiente (IEP alto) en meses de mayor recaudación. Posible efecto de **economías de escala** |

### 🔵 Correlaciones Débiles o Nulas (|r| < 0.30)

| Par de Variables | r | Interpretación |
|---|---|---|
| **Recaudo_Neto ↔ IPC** | **−0.188** | La inflación **no afecta directamente** el recaudo en el mismo periodo. La relación es leve e inversa |
| **Recaudo_Neto ↔ Salario_Minimo** | **−0.182** | Los incrementos salariales no se reflejan inmediatamente en el recaudo |
| **Recaudo_Neto ↔ UPC** | **−0.118** | La variación de la UPC tiene influencia mínima sobre el nivel de recaudo mensual |
| **lag_12 ↔ IPC** | **−0.210** | Correlación inversa débil: periodos de alta inflación no se asocian con mayor recaudo histórico |

---

## 3. Diagnóstico Estadístico

### ⚠️ Multicolinealidad Detectada

> [!WARNING]
> **Salario_Minimo y UPC** tienen **r = 0.903**, lo cual indica **multicolinealidad severa**. Incluir ambas variables simultáneamente como exógenas en SARIMAX provocará:
> - Inflación artificial de errores estándar
> - Coeficientes inestables y difíciles de interpretar
> - Posible no convergencia del optimizador

**Recomendación:** Usar **solamente UPC** como variable exógena en SARIMAX (es la más relevante para rentas de salud), o bien crear un **índice compuesto** mediante Análisis de Componentes Principales (PCA).

### ✅ Independencia Confirmada

Las variables macro (IPC, Salario, UPC) muestran correlaciones **débiles** con Recaudo_Neto directo (|r| < 0.20). Esto sugiere que:

1. Las variables macro operan con **rezago** — su efecto se materializa meses después (ajustes tarifarios, indexación diferida).
2. El recaudo está más determinado por **estacionalidad propia** (lag_12 = 0.875) que por factores macro contemporáneos.
3. Un modelo SARIMAX con variables macro **rezagadas** (lag 1-3 meses) podría capturar mejor el efecto diferido.

---

## 4. Implicaciones para los Modelos de Pronóstico

### Modelo SARIMA (Notebook 04)
- ✅ La alta autocorrelación (lag_12 = 0.875) confirma que **SARIMA es apropiado** como modelo base.
- El componente estacional (s=12) capturará la mayor parte de la varianza explicada.

### Modelo SARIMAX (Notebook 05)
- ⚠️ **No incluir** Salario_Minimo y UPC simultáneamente (r = 0.903).
- **Recomendación:** Usar IPC y UPC como exógenas, **con rezagos de 1-3 meses**.
- El IEP puede servir como variable de control para medir la eficiencia del portafolio.

### Modelo Prophet (Notebook 06)
- Las variables macro son candidatas a **regresores adicionales** en Prophet.
- Prophet maneja mejor la multicolinealidad que SARIMAX (regularización bayesiana).
- Incluir IPC, UPC y YoY_Recaudo como regresores.

### Modelo XGBoost (Notebook 07)
- XGBoost tolera multicolinealidad (no afecta árboles de decisión).
- **Incluir todas las variables** + sus rezagos como features.
- La importancia de features (SHAP) revelará cuáles contribuyen realmente.

### Modelo LSTM (Notebook 08)
- Las correlaciones débiles con macro confirman que LSTM debe enfocarse en **patrones secuenciales** (lag_12 como feature dominante).
- Normalización obligatoria por las diferentes escalas entre variables.

---

## 5. Recomendaciones para la Tesis

### 5.1 Redacción Sugerida (Capítulo de Resultados)

> *"El análisis de correlación revela que el recaudo de rentas cedidas presenta una fuerte autocorrelación anual (r = 0.875 con lag-12), lo cual confirma la dominancia del componente estacional en la serie. Las variables macroeconómicas (IPC, Salario Mínimo, UPC) muestran correlaciones contemporáneas débiles con el recaudo (|r| < 0.20), sugiriendo que su influencia opera con rezago temporal. Se identificó multicolinealidad significativa entre Salario Mínimo y UPC (r = 0.903), lo cual fue considerado en la especificación de los modelos SARIMAX."*

### 5.2 Tabla Resumen para el Documento

| Relación | r | Fuerza | Implicación para Modelos |
|---|---|---|---|
| Recaudo ↔ lag_12 | 0.875 | Fuerte | SARIMA como modelo base |
| Recaudo ↔ YoY | 0.430 | Moderada | Feature en XGBoost/LSTM |
| Recaudo ↔ IEP | 0.370 | Moderada | Indicador KPI financiero |
| Recaudo ↔ IPC | −0.188 | Débil | Efecto diferido, usar con rezago |
| Recaudo ↔ Salario | −0.182 | Débil | Multicolineal con UPC — excluir |
| Recaudo ↔ UPC | −0.118 | Débil | Usar como exógena con rezago en SARIMAX |

### 5.3 Conclusión del Análisis

> [!IMPORTANT]
> La matriz de correlación demuestra que el **principal predictor del recaudo** es su propia historia (lag-12), no las variables macroeconómicas contemporáneas. Esto tiene dos implicaciones:
> 1. Los modelos puramente estacionales (SARIMA) tendrán buen desempeño base.
> 2. El valor añadido de variables exógenas (SARIMAX, Prophet) provendrá de **capturar efectos diferidos** y **shocks** que la estacionalidad sola no explica.

---

## 6. Análisis de Dispersión: Recaudo vs Variables Macroeconómicas

Los **scatter plots** con línea de regresión lineal muestran la relación bivariada entre el Recaudo Neto (eje Y, en escala ×10¹¹ COP) y cada variable exógena. Los puntos están coloreados por **año** (escala viridis: amarillo = más reciente, azul oscuro = más antiguo), lo cual revela la evolución temporal de cada relación.

---

### 6.1 Panel Superior Izquierdo: Recaudo lag-12 (r = 0.875)

**Patrón visual:** Nube de puntos alineada con pendiente positiva clara. La línea de regresión (roja discontinua) captura bien la tendencia.

**Lectura:**
- Es la relación **más fuerte y confiable**. Cada peso recaudado hace 12 meses predice ~0.87 pesos del recaudo actual.
- Los puntos **amarillos** (años recientes: 2024–2025) se ubican en la **esquina superior derecha**, lo cual indica **crecimiento sostenido** — la serie tiene tendencia positiva.
- Los puntos **azules** (2021–2022) se concentran en la **esquina inferior izquierda**, confirmando niveles de recaudo más bajos al inicio del periodo.
- **Dispersión creciente**: a medida que el recaudo crece, la varianza también aumenta → señal de **heterocedasticidad** que justifica el uso de transformación logarítmica o modelos multiplicativos.

> [!TIP]
> Este panel confirma que la **estacionalidad anual es el principal driver** del recaudo. Un modelo SARIMA(p,d,q)(P,D,Q)[12] capturará la mayor parte de esta relación.

---

### 6.2 Panel Superior Derecho: Inflación — IPC (r = −0.188)

**Patrón visual:** Nube dispersa sin tendencia clara. La línea de regresión es casi **horizontal** con leve pendiente negativa.

**Lectura:**
- La inflación **no explica** el recaudo de forma directa ni contemporánea.
- Los **puntos amarillos** (2024–2025, IPC ~4.5–5.5%) se dispersan verticalmente desde $1.5×10¹¹ hasta $5×10¹¹, mostrando que con baja inflación se dan tanto recaudos altos como bajos.
- Los **puntos verdes** (2023, IPC ~9.3%) y **azules** (2022, IPC ~13.1%) muestran que periodos de **alta inflación** no necesariamente generaron mayor recaudo.
- El punto **amarillo extremo** (~$5×10¹¹ con IPC ~5.5%) es probablemente un **pico estacional de enero/julio** — un outlier que no se explica por inflación sino por calendario fiscal.

**Conclusión:** La inflación opera con **rezago** de 6-12 meses. Los ajustes tarifarios basados en IPC se aplican al inicio del año siguiente, no en el mismo periodo de variación inflacionaria.

---

### 6.3 Panel Inferior Izquierdo: Δ Salario Mínimo (r = −0.182)

**Patrón visual:** Nube muy dispersa con tendencia ligeramente negativa. Se observan **clusters verticales** por año (ya que el salario mínimo cambia una vez al año).

**Lectura:**
- Cada **columna vertical** de puntos corresponde a un año: los puntos se alinean verticalmente porque el Δ Salario Mínimo es **constante durante los 12 meses del año** (se fija en enero).
- **2023** (Δ = 16%) tiene los puntos más a la derecha — fue el mayor incremento salarial del periodo, pero el recaudo asociado varía enormemente ($1.7×10¹¹ hasta $4×10¹¹).
- La pendiente negativa débil es **espuria**: no significa que más salario cause menos recaudo, sino que los años con mayor incremento salarial (2023) coincidieron con niveles de recaudo aún en crecimiento, mientras que los años recientes (2025, mayor recaudo) tuvieron incrementos salariales menores.

> [!NOTE]
> La estructura de **clusters anuales** invalida el supuesto de observaciones independientes requerido por la correlación de Pearson. Se recomienda usar **correlación de Spearman** o analizar la relación a nivel anual (no mensual) para obtener una medida más confiable.

---

### 6.4 Panel Inferior Derecho: Δ UPC (r = −0.118)

**Patrón visual:** Similar al salario mínimo — clusters verticales anuales con línea de regresión casi plana.

**Lectura:**
- La UPC, al igual que el salario, se ajusta **una vez al año**, generando los mismos clusters verticales.
- Con r = −0.118, la relación es **prácticamente inexistente** a nivel contemporáneo.
- Los **puntos amarillos** (2025, Δ UPC ~8%) tienen el rango de recaudo más amplio, sugiriendo que otros factores (estacionalidad, volumen de transacciones) dominan sobre la UPC.
- La **paradoja aparente** de pendiente negativa (mayor UPC → menos recaudo) se explica porque 2022–2023 tuvieron los mayores incrementos de UPC pero el recaudo general era más bajo al inicio del periodo analizado. Es un efecto de **confusión temporal**, no causal.

---

### 6.5 Diagnóstico Conjunto de los Scatter Plots

| Variable Exógena | r | Linealidad | Clusters | Heterocedasticidad | Uso Recomendado |
|---|---|---|---|---|---|
| **lag_12** | 0.875 | ✅ Alta | No | ⚠️ Sí (creciente) | Feature principal en todos los modelos |
| **IPC** | −0.188 | ❌ Baja | No | ❌ No | Usar con **rezago 6-12 meses** en SARIMAX |
| **Salario_Minimo** | −0.182 | ❌ Baja | ⚠️ Anuales | ❌ No | **Excluir** de SARIMAX (multicolineal con UPC) |
| **UPC** | −0.118 | ❌ Baja | ⚠️ Anuales | ❌ No | Usar como proxy anual con **rezago** |

> [!IMPORTANT]
> **Hallazgo metodológico**: Las variables macro anuales (Salario, UPC) generan **clusters artificiales** en la dispersión mensual. Esto viola el supuesto de independencia de observaciones. Para la tesis se recomienda:
> 1. Reportar tanto **Pearson** (lineal) como **Spearman** (monótona) para robustecer el análisis.
> 2. Complementar con **correlación cruzada** (CCF) para detectar rezagos óptimos.
> 3. Para XGBoost, incluir la variable **como feature categórica anual**, no como valor numérico repetido mensualmente.

---

### 6.6 Redacción Sugerida para la Tesis (Sección de Scatter Plots)

> *"Los diagramas de dispersión (Figura N) revelan patrones diferenciados entre las variables exógenas analizadas. El recaudo lag-12 muestra una relación lineal positiva fuerte (r = 0.875) con evidencia de heterocedasticidad creciente, consistente con la tendencia alcista de la serie. En contraste, las variables macroeconómicas — IPC (r = −0.188), Salario Mínimo (r = −0.182) y UPC (r = −0.118) — exhiben relaciones contemporáneas débiles con el recaudo. Se observa que las variables de ajuste anual (Salario Mínimo y UPC) generan clusters de observaciones mensuales que comparten el mismo valor exógeno, lo cual limita la validez del coeficiente de correlación lineal. La codificación cromática por año evidencia una tendencia de crecimiento temporal del recaudo independiente de las fluctuaciones macroeconómicas, reforzando la hipótesis de que la estacionalidad fiscal propia constituye el principal determinante del comportamiento del recaudo de rentas cedidas (ver Tabla N)."*

---

## 7. Índice de Eficiencia del Portafolio (IEP)

### 7.1 Definición y Fórmula

El **IEP** (Índice de Eficiencia del Portafolio) es un indicador construido en este estudio para medir si el crecimiento del recaudo supera o no el crecimiento de los costos regulados:

$$
\text{IEP}_t = \frac{\text{YoY\_Recaudo}_t}{\Delta\text{UPC}_t}
$$

Donde:
- **YoY_Recaudo** = variación porcentual interanual del recaudo neto mensual: $(R_t - R_{t-12}) / R_{t-12}$
- **Δ UPC** = variación porcentual anual de la Unidad de Pago por Capitación

### 7.2 Interpretación

| Valor IEP | Significado | Color en gráfica |
|---|---|---|
| **IEP > 1** | El recaudo crece **más rápido** que los costos → Portafolio **eficiente** | 🟢 Verde |
| **IEP = 1** | Crecimiento del recaudo **iguala** al de los costos → **Equilibrio** | ⚫ Línea punteada |
| **0 < IEP < 1** | El recaudo crece, pero **menos** que los costos → **Pérdida relativa** | 🔴 Rojo |
| **IEP < 0** | El recaudo **disminuye** mientras los costos suben → **Déficit real** | 🔴 Rojo |

---

### 7.3 Lectura de la Gráfica de Barras del IEP

La gráfica muestra barras mensuales del IEP con la **línea de equilibrio** (IEP = 1) como referencia punteada.

**Patrones observados:**

1. **Barras verdes dominantes (IEP > 1):** La mayoría de meses el recaudo creció más rápido que la UPC. Esto indica que el departamento del Quibdó ha logrado un recaudo **por encima del ajuste inflacionario** de costos, señal positiva de gestión de rentas.

2. **Barras rojas esporádicas (IEP < 1 o negativo):** Meses puntuales donde el crecimiento del recaudo fue inferior al incremento de costos. Estos periodos coinciden típicamente con:
   - Meses de **bajo recaudo estacional** (febrero, septiembre)
   - Rezagos administrativos en la facturación
   - Eventos atípicos como cambios regulatorios o retrasos en pagos

3. **Volatilidad alta del IEP:** Las barras varían ampliamente de un mes a otro. Esto refleja la **naturaleza estacional** del recaudo — los picos de enero y julio generan IEP muy altos, mientras que los meses valle comprimen el indicador.

4. **Tendencia temporal:** Si se observa una **mayor proporción de barras verdes** hacia el final del periodo (2024–2025), esto sugiere una **mejora progresiva** en la eficiencia del portafolio de rentas.

---

### 7.4 Estadísticas del IEP

El notebook calcula las siguientes métricas resumen:

- **IEP Promedio**: Valor medio del IEP en todo el periodo. Un valor > 1 indica que, en promedio, el recaudo ha superado el crecimiento de costos.
- **Meses eficientes (IEP > 1)**: Proporción de meses donde el portafolio fue eficiente respecto al total. Una proporción superior al 60% se considera **gestión positiva**.

---

### 7.5 Implicaciones para la Tesis

> [!IMPORTANT]
> El IEP es un **indicador original** propuesto en esta investigación. Su principal valor para la tesis es:
> 1. **Aporta valor diferenciador**: No es un indicador estándar de la literatura de finanzas públicas colombiana, lo cual puede constituir una **contribución metodológica**.
> 2. **Conecta modelación con decisión política**: Permite evaluar si los modelos de pronóstico generan proyecciones que implican escenarios de eficiencia o déficit.
> 3. **Facilita comunicación con tomadores de decisión**: Un director de rentas entiende más fácilmente "el recaudo creció 1.3 veces más rápido que los costos" que "el p-value del coeficiente UPC en SARIMAX fue 0.043".

### 7.6 Limitaciones del IEP

- **Sensibilidad a división por cero**: Si Δ UPC = 0 (sin incremento de costos), el IEP es indefinido. Se resuelve excluyendo esos periodos o usando un valor mínimo de Δ UPC.
- **Amplificación de valores extremos**: Cuando Δ UPC es muy pequeño, el IEP se dispara. Se recomienda reportar la **mediana** además del promedio.
- **No captura eficiencia absoluta**: Un IEP > 1 no significa necesariamente que el recaudo sea "suficiente", solo que creció más rápido que los costos regulados.

---

### 7.7 Redacción Sugerida para la Tesis (Sección del IEP)

> *"Se construyó un Índice de Eficiencia del Portafolio (IEP) definido como la razón entre la variación interanual del recaudo neto y la variación anual de la UPC. Este indicador permite evaluar si el crecimiento del recaudo supera el incremento de los costos regulados por el sistema de salud. El análisis mensual del periodo 2021–2025 evidencia que el IEP promedio fue [valor], con [N] de [total] meses mostrando valores superiores a 1 (eficiencia). La distribución temporal del IEP revela estacionalidad pronunciada, con picos de eficiencia en los meses de mayor facturación (enero, julio) y valles en meses de recaudo bajo. Este indicador constituye una herramienta complementaria para la evaluación de la gestión de rentas cedidas, al vincular directamente el desempeño recaudatorio con la dinámica de costos del sistema (ver Figura N)."*

---

## 8. Conclusión General del Análisis de Correlación

El análisis correlacional revela que las **variables macroeconómicas contemporáneas tienen bajo poder explicativo** sobre el recaudo mensual de rentas cedidas. Los tres hallazgos principales son:

1. **La autoestacionalidad domina**: El recaudo lag-12 (r = 0.875) es el predictor más fuerte, confirmando que el comportamiento histórico propio de la serie es su mejor explicación.

2. **Las variables macro operan con rezago**: IPC, UPC y Salario Mínimo muestran correlaciones débiles con el recaudo contemporáneo, pero su efecto probable opera 6-12 meses después a través de ajustes tarifarios anuales.

3. **Multicolinealidad exógena**: Salario Mínimo y UPC (r = 0.903) no pueden usarse simultáneamente como regresores. Se recomienda usar solo UPC como la variable más relevante para el sector salud.

Estas conclusiones fundamentan la decisión de utilizar modelos **estacionales puros** (SARIMA) como línea base, complementados con modelos que incorporen variables exógenas rezagadas (SARIMAX con UPC lag-12) y modelos de machine learning (XGBoost) que capturen relaciones no lineales.

---

*Documento generado como parte del análisis del Notebook 03_Correlacion_Macro.ipynb*
*Sistema de Análisis de Rentas Cedidas — Tesis de Maestría*
