# 📊 Funciones de Autocorrelación (ACF / PACF) — Recaudo de Rentas Cedidas

**Documento de soporte para tesis de maestría**
*Serie: Recaudo Mensual Neto | Periodo: Oct 2021 – Dic 2025*

---

## 1. ¿Qué son ACF y PACF?

| Función | Qué mide | Para qué sirve |
|---------|----------|-----------------|
| **ACF** (Autocorrelación) | Correlación entre la serie y sus rezagos, **incluyendo** efectos indirectos | Determinar el orden **q** del componente MA |
| **PACF** (Autocorrelación Parcial) | Correlación entre la serie y sus rezagos, **eliminando** efectos intermedios | Determinar el orden **p** del componente AR |

- **Banda sombreada rosa:** Intervalo de confianza al 95%. Si una barra queda **dentro** de la banda, el rezago **no es significativo**.
- **Barras que sobresalen:** Rezagos estadísticamente significativos (p < 0.05).

---

## 2. Lectura de los 4 Paneles

### 2.1 ACF — Nivel (serie original, sin transformar)

**Observaciones:**
- **Lag 1:** Barra negativa significativa (~-0.35) → el mes siguiente tiende a moverse en dirección **opuesta**
- **Lag 6:** Pico positivo significativo (~+0.50) → correlación fuerte cada **6 meses** → confirma ciclo semestral
- **Lag 12:** Pico positivo significativo (~+0.60) → correlación fuerte cada **12 meses** → confirma ciclo anual
- **Lags 2-5, 7-11:** Dentro de la banda o marginales → no significativos individualmente
- **Decaimiento lento:** Las barras en lags 6 y 12 NO decaen rápidamente → indica que la serie **no es estacionaria** en nivel

**Diagnóstico:** Serie con estacionalidad clara (s=6 y s=12) y probable no estacionariedad.

### 2.2 PACF — Nivel

**Observaciones:**
- **Lag 1:** Negativo significativo (~-0.35)
- **Lag 6:** Positivo significativo (~+0.45) → efecto directo del semestre anterior
- **Lag 12:** Positivo significativo (~+0.30) → efecto directo del año anterior
- **Demás lags:** Dentro de la banda → no tienen efecto parcial significativo

**Diagnóstico:** Solo los lags 1, 6 y 12 tienen efecto **directo** sobre el valor actual. Los demás son efectos indirectos propagados.

### 2.3 ACF — Primera Diferencia (Δyₜ = yₜ - yₜ₋₁)

**Observaciones:**
- **Lag 1:** Barra negativa muy fuerte (~-0.55) → sobrediferenciación posible, o componente MA(1) fuerte
- **Lag 5:** Pico positivo significativo (~+0.50) → el ciclo semestral persiste aún después de diferenciar
- **Lag 6:** Positivo marginalmente significativo
- **Lag 11-12:** Picos positivos significativos (~+0.35 y ~+0.60) → el ciclo anual también persiste
- **Lag 23-24:** Barras significativas ~ segundo ciclo anual

**Diagnóstico:** La primera diferencia **NO elimina la estacionalidad**. Se necesita **diferenciación estacional** adicional (D=1).

### 2.4 PACF — Primera Diferencia

**Observaciones:**
- **Lag 1:** Negativo muy fuerte (~-0.55) → componente AR(1) significativo
- **Lag 5:** Negativo significativo (~-0.25)
- **Lag 11-12:** Barras marginalmente significativas
- **Resto:** Dentro de la banda

**Diagnóstico:** Confirma que tras diferenciar, un modelo AR(1) captura la mayor parte de la dependencia lineal, pero persiste la estructura estacional.

---

## 3. Tabla Resumen de Rezagos Significativos

| Lag | ACF Nivel | PACF Nivel | ACF 1ª Dif | PACF 1ª Dif | Interpretación |
|-----|-----------|------------|------------|-------------|----------------|
| 1 | -0.35* | -0.35* | -0.55* | -0.55* | Efecto rebote mes a mes (sube→baja) |
| 6 | +0.50* | +0.45* | +0.50* | — | **Ciclo semestral** (Ene↔Jul) |
| 12 | +0.60* | +0.30* | +0.60* | — | **Ciclo anual** (Ene↔Ene, Jul↔Jul) |
| 24 | — | — | +0.35* | — | Segundo armónico anual |

*\* = estadísticamente significativo (fuera de banda 95%)*

---

## 4. Diagnóstico para Modelado SARIMAX

De los 4 paneles se extraen los **órdenes del modelo SARIMAX(p,d,q)(P,D,Q)[s]**:

### 4.1 Componente no estacional (p, d, q)

| Parámetro | Valor sugerido | Evidencia |
|-----------|---------------|-----------|
| **d = 1** | Una diferencia regular | La ACF a nivel no decae → serie no estacionaria. La primera diferencia reduce dependencia |
| **p = 1** | AR(1) | PACF de primera diferencia muestra solo lag 1 significativo |
| **q = 1** | MA(1) | ACF de primera diferencia muestra corte abrupto tras lag 1 |

### 4.2 Componente estacional (P, D, Q)[s]

| Parámetro | Valor sugerido | Evidencia |
|-----------|---------------|-----------|
| **s = 12** | Periodo anual | Picos en lag 12 tanto en ACF como PACF |
| **D = 1** | Una diferencia estacional | Los picos estacionales persisten tras d=1 → se necesita diferenciación estacional |
| **P = 1** | SAR(1) | PACF a nivel: lag 12 significativo individualmente |
| **Q = 1** | SMA(1) | ACF a nivel: lag 12 significativo, patrón de corte |

### 4.3 Modelo inicial recomendado

```
SARIMAX(1,1,1)(1,1,1)[12] + exógenas
```

> Este es el modelo "airline" clásico de Box-Jenkins con extensión para variables exógenas (IPC_Idx). Se valida con `auto_arima` de pmdarima y criterios AIC/BIC. El notebook `04_SARIMAX.ipynb` implementa la búsqueda automática de órdenes.

---

## 5. Recomendaciones para la Tesis de Maestría

### 5.1 Sobre la identificación del modelo

| Recomendación | Justificación |
|---------------|---------------|
| Usar `auto_arima` de `pmdarima` para búsqueda automática | Confirma o mejora la identificación visual ACF/PACF |
| Reportar **AIC, BIC y HQIC** del modelo seleccionado | Estándar académico para comparación de modelos |
| Comparar SARIMAX(1,1,1)(1,1,1)[12] vs variantes | Probar (2,1,1)(1,1,1)[12] y (1,1,2)(1,1,1)[12] como alternativas |

### 5.2 Sobre la estacionalidad dual

| Hallazgo | Implicación |
|----------|-----------|
| Ciclo s=6 visible en ACF (lag 6 significativo) | Considerar modelo con **s=6** además de s=12 |
| Pero s=12 es más fuerte (lag 12 > lag 6) | El modelo principal debe usar **s=12** |
| Alternativa: SARIMA con s=12 + dummy semestral | Captura ambos ciclos sin duplicar estacionalidad |

### 5.3 Sobre los diagnósticos post-modelo

Después de ajustar el SARIMA, verificar que los **residuos** cumplan:

| Prueba | Criterio de éxito | Herramienta |
|--------|-------------------|-------------|
| ACF de residuos | Ningún lag significativo (ruido blanco) | `plot_diagnostics()` |
| Ljung-Box | p-valor > 0.05 | `acorr_ljungbox()` |
| Normalidad | Shapiro-Wilk p > 0.05 o Q-Q plot lineal | `stats.shapiro()` |
| Homocedasticidad | Varianza constante en el tiempo | Gráfico residuos vs tiempo |

### 5.4 Sobre la presentación en la tesis

> [!TIP]
> **Texto sugerido para la sección de metodología:**
>
> "El análisis de las funciones de autocorrelación (ACF) y autocorrelación parcial (PACF) reveló un patrón estacional dominante con periodo s=12, evidenciado por picos significativos en los rezagos 6 y 12 de la ACF a nivel (r₆ ≈ 0.50, r₁₂ ≈ 0.60). La persistencia de estos picos tras la primera diferencia regular (d=1) confirmó la necesidad de diferenciación estacional (D=1). El corte abrupto en el lag 1 de la PACF diferenciada (φ₁ ≈ -0.55) sugirió un componente autorregresivo AR(1), mientras que el comportamiento análogo en la ACF indicó un componente de media móvil MA(1). El modelo inicial identificado fue SARIMA(1,1,1)(1,1,1)[12], posteriormente validado mediante búsqueda automática con el criterio AIC."

---

## 6. Diagrama de Decisión

```
Serie Original
    │
    ├── ACF no decae → d ≥ 1 (no estacionaria)
    │
    └── Diferenciar (d=1)
         │
         ├── ACF lag 12 aún significativo → D = 1 (estacionalidad persiste)
         │
         ├── PACF lag 1 significativo → p = 1
         │
         ├── ACF lag 1 corte abrupto → q = 1
         │
         ├── PACF lag 12 significativo → P = 1
         │
         └── ACF lag 12 corte → Q = 1
              │
              └── SARIMA(1,1,1)(1,1,1)[12] ✓
```
