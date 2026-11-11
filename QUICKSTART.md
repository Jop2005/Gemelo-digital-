# Quick Start - Backend GraphQL

## 🚀 Inicio Rápido

**IMPORTANTE**: Todos los comandos deben ejecutarse desde el directorio `backend/`

### Opción 1: Usar el script (Recomendado)

**Linux/macOS:**
```bash
cd backend
./run.sh
```

**Windows:**
```bash
cd backend
run.bat
```

### Opción 2: Comando manual

**Linux/macOS:**
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Windows:**
```bash
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Opción 3: Python directo

**Desde el directorio raíz del proyecto:**
```bash
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
python -m app.main
```

## 🌐 URLs Disponibles

Una vez iniciado el servidor:

- **API Root**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **GraphQL Playground**: http://localhost:8000/graphql
- **Panel Classifier API**: http://localhost:8000/api/classify-panel (POST)

## 🧪 Probar el API

### En el GraphQL Playground

Abre http://localhost:8000/graphql en tu navegador y prueba esta query:

```graphql
query {
  hello
}
```

Deberías ver:

```json
{
  "data": {
    "hello": "Hello from Digital Twin GraphQL API!"
  }
}
```

### Usando curl

```bash
curl -X POST http://localhost:8000/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "{ hello }"}'
```

### Clasificar imagen de panel solar

```bash
curl -X POST http://localhost:8000/api/classify-panel \
  -F "file=@path/to/panel_image.jpg"
```

Respuesta esperada:
```json
{
  "clasificacion": "limpio",
  "porcentaje_limpio": 85.23,
  "porcentaje_sucio": 14.77
}
```

### Predicciones con Machine Learning

El backend incluye un modelo Random Forest entrenado para predecir la producción solar basándose en datos meteorológicos de Open-Meteo.

**1. Predecir para fechas/horas específicas:**

```graphql
query {
  mlPredict(datetimes: ["2025-01-15T13:00:00", "2025-01-15T14:00:00"]) {
    datetime
    productionKw
    weather {
      temperature2m
      relativeHumidity2m
      windSpeed10m
      cloudCover
      shortwaveRadiation
    }
  }
}
```

**2. Predecir las próximas 24 horas:**

```graphql
query {
  mlPredictNextHours(hours: 24) {
    datetime
    productionKw
    weather {
      temperature2m
      cloudCover
      shortwaveRadiation
    }
  }
}
```

**3. Predecir un rango de fechas:**

```graphql
query {
  mlPredictDateRange(
    startDate: "2025-01-15"
    endDate: "2025-01-17"
  ) {
    datetime
    productionKw
    weather {
      shortwaveRadiation
      cloudCover
    }
  }
}
```

**4. Ver información del modelo:**

```graphql
query {
  mlModelInfo {
    loaded
    modelName
    testRmse
    testR2
    testMae
    features
    trainingDate
  }
}
```

Respuesta esperada:
```json
{
  "data": {
    "mlModelInfo": {
      "loaded": true,
      "modelName": "Random Forest",
      "testRmse": 145.32,
      "testR2": 0.9845,
      "testMae": 98.21,
      "features": [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "cloud_cover",
        "shortwave_radiation",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos"
      ],
      "trainingDate": "2025-01-12 15:30:00"
    }
  }
}
```

## 📝 Notas

- El servidor se recarga automáticamente cuando detecta cambios en el código
- MongoDB debe estar ejecutándose en `localhost:27017`
- Los logs del servidor aparecen en la terminal
- **Para usar el clasificador de paneles y predicciones ML**, instala las dependencias actualizadas:
  ```bash
  cd backend
  source venv/bin/activate  # Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```
- El modelo de ML se carga automáticamente al iniciar el servidor (puede tardar unos segundos)
- **Para entrenar el modelo de predicción**, ejecuta el notebook:
  ```bash
  cd backend/notebooks
  jupyter notebook solar_production_prediction.ipynb
  ```
  El modelo entrenado se guardará automáticamente en `backend/models/`

## 🛑 Detener el Servidor

Presiona `Ctrl + C` en la terminal donde está corriendo el servidor.
