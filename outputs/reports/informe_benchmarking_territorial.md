======================================================================
INFORME DE BENCHMARKING MULTIDIMENSIONAL TERRITORIAL
Sistema de Analisis de Rentas Cedidas — ADRES
======================================================================

1. CONCENTRACION FISCAL
   - Indice de Gini: 0.9465 (Alta concentracion)
   - Top 5 entidades: 47.6% del recaudo total
   - Top 10 entidades: 61.0% del recaudo total
   - Pareto (20% entidades): 96.6% del recaudo
   - Validacion Orozco-Gallo (2015): CONFIRMADA

2. TIPOLOGIAS TERRITORIALES (K-Means, k=4)
   Consolidados:
     - Entidades: 38
     - Recaudo mediana mensual: $3,365 millones
     - CV mediana: 45.7%
     - Participacion en recaudo total: 88.9%
   Emergentes:
     - Entidades: 280
     - Recaudo mediana mensual: $22 millones
     - CV mediana: 133.0%
     - Participacion en recaudo total: 6.1%
   Dependientes:
     - Entidades: 230
     - Recaudo mediana mensual: $21 millones
     - CV mediana: 93.0%
     - Participacion en recaudo total: 3.4%
   Criticos:
     - Entidades: 552
     - Recaudo mediana mensual: $5 millones
     - CV mediana: 136.7%
     - Participacion en recaudo total: 1.6%

3. ASIMETRIA ESTRUCTURAL (Bogota vs Choco)
   - Ratio de desigualdad mediano: 18x
   - Bogota concentra ~48% del recaudo con 1 entidad (FFDS)
   - Patron estacional: diferenciado

4. DEFLACTACION POR IPC
   - Efecto inflacion sobre recaudo nominal: 10.1%
   - Crecimiento real acumulado: -35.3%

5. SISTEMA DE ALERTA TEMPRANA (SAT)
   VERDE: 0 entidades (0.0%)
   AMARILLO: 4 entidades (0.4%)
   NARANJA: 19 entidades (1.7%)
   ROJO: 1077 entidades (97.9%)
   Total en riesgo (Naranja + Rojo): 1096 entidades (99.6%)

6. RECOMENDACIONES
   a) Implementar monitoreo trimestral con semaforo SAT para entidades
      clasificadas en NARANJA y ROJO.
   b) Establecer fondos de estabilizacion para entidades Dependientes
      con CV > 30%, que representen riesgo de desfinanciamiento.
   c) Fortalecer capacidad de gestion fiscal en entidades Criticas
      mediante asistencia tecnica focalizada.
   d) La deflactacion por IPC debe ser obligatoria en la
      evaluacion de desempeno fiscal para aislar crecimiento real.
   e) El modelo XGBoost (MAPE 5.05%) debe integrarse al SAT como
      herramienta predictiva para anticipar caidas en el recaudo.

======================================================================