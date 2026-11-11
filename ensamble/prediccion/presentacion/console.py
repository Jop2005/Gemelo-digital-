import numpy as np
from prediccion.presentacion.logger import get_logger
from prediccion.presentacion.validators import normalizar_ruta, leer_numero

class Console:
    
    def __init__(self):
        self.logger = get_logger()
    
    def mostrar_menu(self, opciones: dict):
        self.logger.info("\n" + "=" * 50)
        for key, descripcion in opciones.items():
            self.logger.info(f"{key}. {descripcion}")
        self.logger.info("=" * 50)
    
    def obtener_opcion(self, mensaje: str = "Opción: ") -> str:
        return input(mensaje).strip()
    
    def obtener_ruta_archivo(self) -> str:
        self.logger.info("\n Ingrese la ruta del archivo (CSV o Excel):")
        archivo = input("Ruta: ").strip()
        return normalizar_ruta(archivo)
    
    def obtener_valores_manuales(self, features, obligatorias):
        datos = {}
        self.logger.info("\n Ingrese los valores:")
        for feature in features:
            es_obligatoria = feature in obligatorias
            if feature == 'Hora':
                valor = leer_numero(feature, obligatorio=es_obligatoria, min_valor=0, max_valor=23)
            elif feature == 'Mes':
                valor = leer_numero(feature, obligatorio=es_obligatoria, min_valor=1, max_valor=12)
            elif feature == 'tipo_dia_bin':
                valor = leer_numero(feature, obligatorio=es_obligatoria, min_valor=0, max_valor=1)
            else:
                valor = leer_numero(feature, obligatorio=es_obligatoria)
            datos[feature] = valor
        return datos
    
    def mostrar_resultado(self, prediccion: float, segmento: int, modelo: str, unidad: str):
        self.logger.info("\n" + "=" * 40)
        self.logger.info("RESULTADO:")
        self.logger.info(f"Predicción:  {prediccion:.2f} {unidad}")
        self.logger.info(f"Segmento:  {segmento}")
        self.logger.info(f"Modelo:  {modelo}")
        self.logger.info("=" * 40)
    
    def mostrar_explicacion_shap(self, data: dict, unidad: str):
        self.logger.info("\n" + "=" * 50)
        self.logger.info("EXPLICACIÓN SHAP DE LA PREDICCIÓN")
        self.logger.info("=" * 50)
        if data is None:
            self.logger.error("No se pudo generar la explicación")
            return
        if 'error' in data:
            self.logger.error(f" {data['error']}")
            return
        pred = data.get('prediction', None)
        if pred is None:
            self.logger.error(f"   Error inesperado. Keys: {list(data.keys())}")
            return
        self.logger.info(f"   Predicción:        {pred:.2f} {unidad}")
        self.logger.info(f"   Valor base:        {data['base_value']:.2f} {unidad}")
        self.logger.info("\n   Contribuciones por característica:")
        self.logger.info(f"   {'Característica':<15s} {'Valor':>10s} {'Contribución':>12s}")
        self.logger.info(f"   {'-'*15} {'-'*10} {'-'*12}")
        contributions = data.get('contributions', {})
        features = data.get('features', {})
        sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, contribucion in sorted_contributions:
            valor = features.get(feature, 'N/A')
            valor_str = f"{valor:.2f}" if isinstance(valor, float) else str(valor)
            direccion = "↑" if contribucion > 0 else "↓"
            self.logger.info(f"   {feature:<15s} {valor_str:>10s} {direccion} {abs(contribucion):.2f} {unidad}")
        self.logger.info("\n Las contribuciones positivas (↑) aumentan la predicción.")
        self.logger.info(" Las contribuciones negativas (↓) disminuyen la predicción.")
        self.logger.info("=" * 50)
    
    def mostrar_resultados_lote(self, resultados, target):
        self.logger.info("\n  RESULTADOS:")
        unidad = target.split('_')[0] if '_' in target else 'unidades'
        for i, res in enumerate(resultados):
            self.logger.info(f"   Fila {i+1}: Predicción={res['prediccion']:.2f} {unidad}, "
                              f"Segmento={res['segmento']}, Modelo={res['modelo']}")
    
    def mostrar_error(self, mensaje: str):
        self.logger.error(f"   ❌ {mensaje}")
    
    def mostrar_exito(self, mensaje: str):
        self.logger.success(f"   ✅ {mensaje}")
    
    def mostrar_info(self, mensaje: str):
        self.logger.info(f"   ℹ️ {mensaje}")
    
    def mostrar_advertencia(self, mensaje: str):
        self.logger.warning(f"   ⚠️ {mensaje}")
    
    def pausar(self):
        input("\nPresione Enter para continuar...")