# Backend FastAPI + GraphQL (Strawberry)

API GraphQL para el Gemelo Digital del sistema fotovoltaico.

## Configuración

1. Crear entorno virtual:
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Configurar variables de entorno:
```bash
cp .env.example .env
# Editar .env con tus configuraciones
```

4. Ejecutar servidor de desarrollo:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## GraphQL Playground

Una vez iniciado el servidor, accede a:
- GraphQL Playground: http://localhost:8000/graphql

## Ejemplo de Query

```graphql
query {
  solarData {
    currentProduction
    currentConsumption
    batteryLevel
    timestamp
  }
}
```
