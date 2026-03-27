======================================================================
INFORME DE BENCHMARKING MULTIDIMENSIONAL TERRITORIAL
Sistema STAR de Analisis de Rentas Cedidas - ADRES
======================================================================

MARCO REGULATORIO
  - Ley 1753 de 2015 (Art. 65): Fortalecimiento fiscal territorial
  - Decreto 2265 de 2017: Distribucion y giro de Rentas Cedidas
  - Ley 715 de 2001 (Art. 44): Competencias territoriales en salud

1. CONCENTRACION FISCAL
   Indice de Gini: 0.9465 (Alta)
   Top 5 entidades: 47.6% del recaudo
   Proxy 5 departamentos: 41.6%
   Pareto (20% entidades): 96.6%
   Bottom 50%: 1.06% (vulnerabilidad extrema)
   Validacion Orozco-Gallo: PARCIAL

2. TIPOLOGIAS TERRITORIALES (K-Means, k=4, CV interanual)
   Consolidados:
     Entidades: 69
     Recaudo mediano anual: $6.97 miles MM
     CV interanual mediano: 11.9%
     % del recaudo total: 93.0%
   Emergentes:
     Entidades: 2
     Recaudo mediano anual: $0.90 miles MM
     CV interanual mediano: 167.8%
     % del recaudo total: 0.0%
   Dependientes:
     Entidades: 430
     Recaudo mediano anual: $0.25 miles MM
     CV interanual mediano: 15.5%
     % del recaudo total: 5.2%
   Criticos:
     Entidades: 600
     Recaudo mediano anual: $0.06 miles MM
     CV interanual mediano: 17.5%
     % del recaudo total: 1.8%

3. ASIMETRIA ESTRUCTURAL
   Ratio Bogota/Choco: 18x (mediana mensual)
   Per capita literatura: $12,500 (Bog) vs $667 (Cho) = 18.7:1
   Correlacion estacional: r = 0.388
   Diagnostico: ERRATICO (justifica Alerta Roja)

4. DEFLACTACION Y ELASTICIDAD
   Efecto inflacion: 10.1% del nominal
   Crecimiento nominal: -21.8% vs Real: -35.3%
   Elasticidad beta vs SMLV: 0.501
   Impacto estimado 23% SMLV (2026): +11.5% en recaudo

5. AUTOCORRELACION Y ANOMALIAS
   ACF Lag-12 agregada: 0.622 (memoria anual FUERTE)
   ACF Lag-1 agregada: -0.135 (mes vencido)
   Entidades con R_Lag12 < 0.3: 31 (anomalias STAR)

6. SISTEMA DE ALERTA TEMPRANA (SAT)
   VERDE: 271 entidades (24.6%)
   AMARILLO: 260 entidades (23.6%)
   NARANJA: 267 entidades (24.3%)
   ROJO: 303 entidades (27.5%)
   Total en riesgo (Naranja+Rojo): 570 (51.8%)
   IEP < 1: 1010 entidades en riesgo de costos

7. RECOMENDACIONES
   a) Implementar SAT trimestral para entidades ROJO/NARANJA
   b) Fondos de estabilizacion para Dependientes (CV > Q75)
   c) Fiscalizacion proactiva: entidades con R_Lag12 < 0.3
   d) Deflactacion IPC obligatoria en evaluacion fiscal
   e) XGBoost (MAPE 5.05%) como motor predictivo del STAR
   f) Monitorizar elasticidad SMLV: beta=0.50 ante +23% en 2026
   g) Alerta Roja permanente para Choco: perfil estacional erratico
     y ratio de desigualdad > 10:1 vs Bogota

======================================================================