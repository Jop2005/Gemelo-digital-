"""
Consola para la UI del pipeline.

Responsabilidad: Mostrar mensajes formateados al usuario.
"""

from logger import get_logger

class Console:
    """
    Vista de consola para el pipeline.
    
    Responsabilidad única: Presentación de mensajes al usuario.
    No contiene lógica de negocio.
    """
    
    def __init__(self):
        self.logger = get_logger()
    
    # ==================== MENÚS ====================
    
    def mostrar_menu(self, opciones: dict) -> None:
        """Muestra un menú con opciones numeradas."""
        self.logger.info("\n" + "=" * 50)
        for key, descripcion in opciones.items():
            self.logger.info(f"{key}. {descripcion}")
        self.logger.info("=" * 50)
    
    def obtener_opcion(self, mensaje: str = "Opción: ") -> str:
        """Obtiene una opción del usuario."""
        return input(mensaje).strip()
    
    # ==================== MENSAJES GENÉRICOS ====================
    
    def mostrar_exito(self, mensaje: str) -> None:
        """Muestra un mensaje de éxito."""
        self.logger.success(f" ✅ {mensaje}")
    
    def mostrar_error(self, mensaje: str) -> None:
        """Muestra un mensaje de error."""
        self.logger.error(f" ❌ {mensaje}")
    
    def mostrar_info(self, mensaje: str) -> None:
        """Muestra un mensaje informativo."""
        self.logger.info(f" ℹ️ {mensaje}")
    
    def mostrar_advertencia(self, mensaje: str) -> None:
        """Muestra un mensaje de advertencia."""
        self.logger.warning(f" ⚠️ {mensaje}")
    
    # ==================== RESULTADOS ESPECÍFICOS ====================
    
    def mostrar_resultado_segmentacion(self, resultado) -> None:
        """Muestra el resultado de la segmentación."""
        if resultado:
            self.mostrar_exito("Segmentación generada")
    
    def mostrar_resultado_entrenamiento(self, nombre: str, resultado) -> None:
        """
        Muestra el resultado del entrenamiento de un modelo.
        
        Args:
            nombre: Nombre del modelo entrenado
            resultado: Instancia de ResultadoEntrenamiento
        """
        if not resultado:
            return
        
        if hasattr(resultado, 'es_modelo_ml'):
            if resultado.es_modelo_ml():
                self.mostrar_exito(f"{nombre} entrenado")
                if resultado.tiene_metricas():
                    self.mostrar_info(f" MAE: {resultado.mae:.4f}")
                    self.mostrar_info(f" RMSE: {resultado.rmse:.4f}")
                    self.mostrar_info(f" R²: {resultado.r2:.4f}")
            else:
                self.mostrar_exito(f"{nombre} entrenado")
        
        elif isinstance(resultado, tuple):
            if len(resultado) >= 3:
                _, _, mae = resultado[0], resultado[1], resultado[2]
                self.mostrar_exito(f"{nombre} entrenado. MAE: {mae:.4f}")
            else:
                self.mostrar_exito(f"{nombre} entrenado")
        
        else:
            self.mostrar_exito(f"{nombre} entrenado")
    
    def mostrar_resultado_evaluacion(self, resultado) -> None:
        """Muestra el resultado de la evaluación."""
        if resultado:
            self.mostrar_exito("Evaluación completada")
            if isinstance(resultado, dict) and 'modelos_evaluados' in resultado:
                modelos = resultado['modelos_evaluados']
                self.mostrar_info(f"Modelos evaluados: {', '.join(modelos)}")
    
    def mostrar_resultado_optimizacion(self, resultado) -> None:
        """Muestra el resultado de la optimización."""
        if resultado:
            self.mostrar_exito("Optimización completada")
    
    def mostrar_resultado_interpretabilidad(self, resultado) -> None:
        """Muestra el resultado de la interpretabilidad."""
        if resultado:
            self.mostrar_exito("Interpretabilidad generada")
    
    # ==================== UTILIDADES ====================
    
    def pausar(self) -> None:
        """Pausa la ejecución hasta que el usuario presione Enter."""
        input("\nPresione Enter para continuar...")