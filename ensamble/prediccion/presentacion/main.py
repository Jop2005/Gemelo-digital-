import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prediccion.acceso_datos.context import DatasetContext
from prediccion.acceso_datos.config import get_config
from prediccion.acceso_datos.joblib_model_repository import JoblibModelRepository
from prediccion.acceso_datos.joblib_imputer_repository import JoblibImputerRepository
from prediccion.acceso_datos.historial_repository import HistorialRepository
from prediccion.acceso_datos.file_reader import FileReader
from prediccion.acceso_datos.config_loader import ConfigLoader
from prediccion.logica_negocio.prediccion_controller import PrediccionController
from prediccion.presentacion.console import Console

class PrediccionUI:
    
    OPCIONES_MENU = {
        '1': 'Predecir desde archivo',
        '2': 'Predecir valores manualmente',
        '3': 'Explicar predicción (SHAP)',
        '4': 'Salir'
    }
    
    def __init__(self, context, model_repo, imputer_repo, historial_repo, file_reader, config_loader):
        self._ui = Console()
        self._features = []
        self._target = ""
        self._obligatorias = []
        try:
            self._controller = PrediccionController(
                context=context,
                model_repo=model_repo,
                imputer_repo=imputer_repo,
                historial_repo=historial_repo,
                file_reader=file_reader,
                config_loader=config_loader
            )
        except Exception as e:
            self._ui.mostrar_error(f"Error al inicializar: {str(e)}")
            self._controller = None
        self._context = context
    
    def run(self):
        if self._controller is None:
            return
        if not self._controller.esta_listo():
            self._ui.mostrar_error("El sistema no está listo. Ejecute primero el pipeline de configuración.")
            return
        
        loader = self._context.get_loader()
        self._features = loader.config.get('features', [])
        self._target = loader.config.get('target', "")
        self._obligatorias = loader.config.get('obligatorias', [])
        
        while True:
            self._ui.mostrar_menu(self.OPCIONES_MENU)
            opcion = self._ui.obtener_opcion()
            
            if opcion == '1':
                self._procesar_prediccion_archivo()
            elif opcion == '2':
                self._procesar_prediccion_manual()
            elif opcion == '3':
                self._procesar_explicacion_shap()
            elif opcion == '4':
                break
            else:
                self._ui.mostrar_error("Opción no válida")
            
            self._ui.pausar()
    
    def _procesar_prediccion_archivo(self):
        archivo = self._ui.obtener_ruta_archivo()
        if not os.path.exists(archivo):
            self._ui.mostrar_error(f"Archivo no encontrado: {archivo}")
            return
        resultados = self._controller.predecir_archivo(archivo, self._features)
        if isinstance(resultados, dict) and 'error' in resultados:
            self._ui.mostrar_error(resultados['error'])
        elif resultados:
            self._ui.mostrar_resultados_lote(resultados, self._target)
        else:
            self._ui.mostrar_error("No se pudieron generar predicciones")
    
    def _procesar_prediccion_manual(self):
        datos = self._ui.obtener_valores_manuales(self._features, self._obligatorias)
        res = self._controller.predecir_manual(datos)
        if isinstance(res, dict) and 'error' in res:
            self._ui.mostrar_error(res['error'])
        elif res:
            unidad = self._target.split('_')[0] if '_' in self._target else 'unidades'
            self._ui.mostrar_resultado(
                res['prediccion'], res['segmento'], res['modelo'], unidad,
            )
        else:
            self._ui.mostrar_error("No se pudo generar la predicción")
    
    def _procesar_explicacion_shap(self):
        datos = self._ui.obtener_valores_manuales(self._features, self._obligatorias)
        self._ui.mostrar_info("\n⏳ Calculando explicación SHAP...")
        explicacion = self._controller.explicar_prediccion(datos, self._features)
        if isinstance(explicacion, dict) and 'error' in explicacion:
            self._ui.mostrar_error(explicacion['error'])
        elif explicacion:
            unidad = self._target.split('_')[0] if '_' in self._target else 'unidades'
            self._ui.mostrar_explicacion_shap(explicacion, unidad)
        else:
            self._ui.mostrar_error("No se pudo generar la explicación")


def main():
    print("Cargando...")
    print("\n" + "=" * 50)
    print("SISTEMA DE PREDICCIÓN")
    print("=" * 50)
    
    dominio = input("Dominio (consumo/generacion): ").strip().lower()
    if dominio not in ['consumo', 'generacion']:
        dominio = 'consumo'
    get_config().set_dominio(dominio)
    
    try:
        if dominio == 'generacion':
            from prediccion.acceso_datos.strategies_generacion import GeneracionStrategy as Strategy
        else:
            from prediccion.acceso_datos.strategies_consumo import ConsumoStrategy as Strategy
        
        strategy = Strategy()
        context = DatasetContext(strategy)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Verifique que el archivo de configuración del dataset existe.")
        return
    
    model_repo = JoblibModelRepository()
    imputer_repo = JoblibImputerRepository()
    historial_repo = HistorialRepository()
    file_reader = FileReader()
    config_loader = ConfigLoader()
    
    ui = PrediccionUI(context, model_repo, imputer_repo, historial_repo, file_reader, config_loader)
    ui.run()


if __name__ == "__main__":
    main()