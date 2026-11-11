import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from console import Console
from configuracion.acceso_datos.context import DatasetContext
from configuracion.acceso_datos.config import get_config
from configuracion.acceso_datos.joblib_model_repository import JoblibModelRepository
from configuracion.acceso_datos.joblib_imputer_repository import JoblibImputerRepository
from configuracion.logica_negocio.fabrica_entrenadores import descubrir_entrenadores_ml
from configuracion.logica_negocio.entrenamiento_knn_controller import EntrenadorKNNController
from configuracion.logica_negocio.pipeline_controller import PipelineController


class PipelineUI:
    
    OPCIONES_MENU = {
        '1': 'Generar segmentación',
        '2': 'Entrenar modelos',
        '3': 'Evaluar modelos',
        '4': 'Optimizar ensamblaje',
        '5': 'Interpretabilidad',
        '6': 'Ejecutar TODO',
        '7': 'Salir'
    }
    
    def __init__(self, context, model_repo, imputer_repo):
        self._context = context
        self._ui = Console()
        self._entrenadores_ml = descubrir_entrenadores_ml(self._context, model_repo)
        self._entrenador_imputer = EntrenadorKNNController(self._context, imputer_repo)
        self._controller = PipelineController(self._context)
    
    def run(self):
        while True:
            self._ui.mostrar_menu(self.OPCIONES_MENU)
            opcion = self._ui.obtener_opcion()
            if opcion == '7':
                self._ui.mostrar_info("¡Hasta luego!")
                break
            self._procesar_opcion(opcion)
            self._ui.pausar()
    
    def _procesar_opcion(self, opcion):
        if opcion == '1':
            self._mostrar_segmentacion()
        elif opcion == '2':
            self._mostrar_entrenamiento()
        elif opcion == '3':
            self._mostrar_evaluacion()
        elif opcion == '4':
            self._mostrar_optimizacion()
        elif opcion == '5':
            self._mostrar_interpretabilidad()
        elif opcion == '6':
            self._mostrar_pipeline_completo()
        else:
            self._ui.mostrar_error("Opción no válida")
    
    def _mostrar_segmentacion(self):
        try:
            self._ui.mostrar_info("Generando segmentación...")
            segmenter = self._controller.ejecutar_segmentacion()
            config = segmenter.get_config()
            tipo = config.get('tipo', 'N/A')
            
            self._ui.mostrar_exito("\n📐 SEGMENTACIÓN GENERADA")
            self._ui.mostrar_info(f"Número de segmentos: {segmenter.n_segments}")
            self._ui.mostrar_info(f"Variables utilizadas: {', '.join(segmenter.variables)}")
            self._ui.mostrar_info(f"Tipo de segmentación: {tipo}")
            
            if tipo == 'rules':
                self._ui.mostrar_info("\nFranjas horarias:")
                self._ui.mostrar_info("  • Madrugada: 0:00 - 5:00")
                self._ui.mostrar_info("  • Mañana:    6:00 - 11:00")
                self._ui.mostrar_info("  • Tarde:     12:00 - 17:00")
                self._ui.mostrar_info("  • Noche:     18:00 - 23:00")
                self._ui.mostrar_info("\nTipos de día:")
                self._ui.mostrar_info("  • Laboral (1)")
                self._ui.mostrar_info("  • Fin de semana / Festivo (0)")
            elif tipo == 'percentiles':
                self._ui.mostrar_info("\nSegmentación por percentiles de radiación solar")
            
            self._ui.mostrar_info("\n📁 Configuración guardada.")
        except Exception as e:
            self._ui.mostrar_error(f"Error al segmentar: {str(e)}")
    
    def _mostrar_entrenamiento(self):
        try:
            self._ui.mostrar_info("Entrenando KNN Imputer...")
            self._controller.ejecutar_entrenamiento_imputer(self._entrenador_imputer)
            self._ui.mostrar_exito("\n🔧 KNN IMPUTER ENTRENADO")
            self._ui.mostrar_info("Guardado en: ./artifacts/imputers/knn_imputer.joblib")
            
            self._ui.mostrar_info("\nEntrenando modelos ML...")
            resultados = self._controller.ejecutar_entrenamiento_ml(self._entrenadores_ml)
            self._ui.mostrar_exito("\n📊 MODELOS ML ENTRENADOS")
            
            mejor_mae = float('inf')
            mejor_modelo = None
            
            for r in resultados:
                res = r['resultado']
                if res.mae < mejor_mae:
                    mejor_mae = res.mae
                    mejor_modelo = r['nombre']
                self._ui.mostrar_info(f"\n🔹 {r['nombre']}")
                self._ui.mostrar_info(f"   MAE:  {res.mae:.2f} kW")
                self._ui.mostrar_info(f"   RMSE: {res.rmse:.2f} kW")
                self._ui.mostrar_info(f"   R²:   {res.r2:.4f} ({res.r2*100:.1f}% de varianza explicada)")
                self._ui.mostrar_info(f"   MAPE: {res.mape:.2f}%")
                self._ui.mostrar_info(f"   MASE: {res.mase:.4f}")
                self._ui.mostrar_info(f"   Parámetros: {res.params}")
            
            self._ui.mostrar_exito(f"\n🏆 Mejor modelo global: {mejor_modelo} (MAE={mejor_mae:.2f} kW)")
            self._ui.mostrar_info("\n📁 Modelos guardados en: ./artifacts/modelos/")
        except Exception as e:
            self._ui.mostrar_error(f"Error al entrenar: {str(e)}")
    
    def _mostrar_cabecera_evaluacion(self, resultado):
        n_seg = resultado.get('n_segments', 0)
        modelos = resultado.get('modelos_evaluados', [])
        self._ui.mostrar_exito("\n📊 EVALUACIÓN POR SEGMENTO")
        self._ui.mostrar_info(f"Segmentos evaluados: {n_seg}")
        self._ui.mostrar_info(f"Modelos evaluados: {', '.join(modelos)}")
        self._ui.mostrar_info(f"Tipo de segmentación: {resultado.get('tipo_segmentacion', 'N/A')}")
    
    def _mostrar_distribucion_muestras(self, resultado):
        distribucion = resultado.get('distribucion', [])
        if distribucion:
            self._ui.mostrar_info("\n📈 Distribución de muestras por segmento:")
            total = sum(distribucion)
            for seg, cantidad in enumerate(distribucion):
                if cantidad > 0:
                    self._ui.mostrar_info(f"   Segmento {seg}: {cantidad} muestras ({cantidad/total*100:.1f}%)")
    
    def _mostrar_tabla_metricas(self, resultado):
        n_seg = resultado.get('n_segments', 0)
        metricas = resultado.get('metricas', {})
        modelos = resultado.get('modelos_evaluados', [])
        
        for modelo in modelos:
            self._ui.mostrar_info(f"\n🔹 {modelo}:")
            self._ui.mostrar_info("   Segmento |   MAE (kW) |   RMSE (kW) |     R²    |   MASE")
            self._ui.mostrar_info("   ---------|------------|-------------|-----------|--------")
            for seg in range(n_seg):
                met = metricas.get(modelo, {}).get(str(seg), {})
                if met:
                    mase = met.get('MASE', float('nan'))
                    mase_str = f"{mase:.4f}" if not np.isnan(mase) else "N/A"
                    self._ui.mostrar_info(
                        f"   {seg:8d} | {met.get('MAE', 0):10.2f} | "
                        f"{met.get('RMSE', 0):11.2f} | {met.get('R2', 0):7.4f} | {mase_str:>6s}"
                    )
    
    def _mostrar_mejores_por_segmento(self, resultado):
        n_seg = resultado.get('n_segments', 0)
        metricas = resultado.get('metricas', {})
        modelos = resultado.get('modelos_evaluados', [])
        
        self._ui.mostrar_info("\n🏆 Mejor modelo por segmento:")
        for seg in range(n_seg):
            mejor, mejor_mae = None, float('inf')
            for modelo in modelos:
                met = metricas.get(modelo, {}).get(str(seg), {})
                if met and met.get('MAE', float('inf')) < mejor_mae:
                    mejor_mae = met['MAE']
                    mejor = modelo
            if mejor:
                self._ui.mostrar_info(f"   Segmento {seg}: {mejor} (MAE={mejor_mae:.2f} kW)")
    
    def _mostrar_evaluacion(self):
        try:
            self._ui.mostrar_info("Evaluando modelos por segmento...")
            resultado = self._controller.ejecutar_evaluacion()
            if isinstance(resultado, dict) and 'error' in resultado:
                self._ui.mostrar_error(resultado['error'])
                return
            if not resultado:
                self._ui.mostrar_error("No se pudo completar la evaluación")
                return
            
            self._mostrar_cabecera_evaluacion(resultado)
            self._mostrar_distribucion_muestras(resultado)
            self._mostrar_tabla_metricas(resultado)
            self._mostrar_mejores_por_segmento(resultado)
            self._ui.mostrar_info("\n📁 Resultados guardados.")
        except Exception as e:
            self._ui.mostrar_error(f"Error al evaluar: {str(e)}")
    
    def _mostrar_optimizacion(self):
        try:
            self._ui.mostrar_info("Optimizando pesos del ensamblaje...")
            resultado = self._controller.ejecutar_optimizacion()
            if isinstance(resultado, dict) and 'error' in resultado:
                self._ui.mostrar_error(resultado['error'])
                return
            if not resultado:
                self._ui.mostrar_error("No se pudo completar la optimización")
                return
            
            segmentos = resultado.get('segmentos', {})
            n_seg = resultado.get('n_segments', 0)
            self._ui.mostrar_exito("\n⚖️ PESOS ÓPTIMOS POR SEGMENTO")
            self._ui.mostrar_info(f"Segmentos optimizados: {n_seg}")
            
            for seg in range(n_seg):
                config = segmentos.get(seg, {})
                modelo = config.get('modelo', 'N/A')
                if modelo == 'Ensemble':
                    self._ui.mostrar_info(f"\n🔹 Segmento {seg}: ENSEMBLE")
                    for nombre, peso in config.get('pesos', {}).items():
                        if peso > 0:
                            self._ui.mostrar_info(f"   • {nombre}: {peso:.3f} ({peso*100:.1f}%)")
                else:
                    self._ui.mostrar_info(f"\n🔹 Segmento {seg}: {modelo} (100%)")
            
            ensambles = sum(1 for c in segmentos.values() if c.get('modelo') == 'Ensemble')
            self._ui.mostrar_info("\n📊 RESUMEN:")
            self._ui.mostrar_info(f"   Segmentos con ensamblaje: {ensambles}/{n_seg}")
            self._ui.mostrar_info(f"   Segmentos con modelo único: {n_seg - ensambles}/{n_seg}")
            self._ui.mostrar_info("\n📁 Configuración guardada.")
        except Exception as e:
            self._ui.mostrar_error(f"Error al optimizar: {str(e)}")
    
    def _mostrar_interpretabilidad(self):
        try:
            self._ui.mostrar_info("Generando reportes de interpretabilidad...")
            resultado = self._controller.ejecutar_interpretabilidad()
            if not resultado:
                self._ui.mostrar_error("No se pudo generar interpretabilidad")
                return
            
            self._ui.mostrar_exito("\n🔍 IMPORTANCIA DE CARACTERÍSTICAS")
            for modelo, importancias in resultado.items():
                self._ui.mostrar_info(f"\n🔹 {modelo}:")
                if not importancias:
                    self._ui.mostrar_info("   No disponible")
                    continue
                total = sum(importancias.values())
                sorted_imp = sorted(importancias.items(), key=lambda x: x[1], reverse=True)
                for variable, importancia in sorted_imp:
                    porcentaje = importancia / total * 100 if total > 0 else 0
                    barra = '█' * int(porcentaje / 5)
                    self._ui.mostrar_info(f"   {variable:12s}: {importancia:.4f} ({porcentaje:5.1f}%) {barra}")
                top_var, top_imp = sorted_imp[0]
                self._ui.mostrar_info(f"\n   📌 Variable más importante para {modelo}: {top_var} ({top_imp/total*100:.1f}%)")
            self._ui.mostrar_info("\n📁 Reportes guardados.")
        except Exception as e:
            self._ui.mostrar_error(f"Error de interpretabilidad: {str(e)}")
    
    def _mostrar_pipeline_completo(self):
        self._ui.mostrar_info("\n🚀 EJECUTANDO PIPELINE COMPLETO")
        self._ui.mostrar_info("=" * 50)
        self._mostrar_segmentacion()
        self._mostrar_entrenamiento()
        self._mostrar_evaluacion()
        self._mostrar_optimizacion()
        self._mostrar_interpretabilidad()
        self._ui.mostrar_exito("\n🎉 PIPELINE COMPLETO FINALIZADO")


def main():
    print("\n" + "=" * 50)
    print("PIPELINE DE CONFIGURACIÓN")
    print("=" * 50)
    
    dominio = input("Dominio (consumo/generacion): ").strip().lower()
    if dominio not in ['consumo', 'generacion']:
        dominio = 'consumo'
    get_config().set_dominio(dominio)
    
    try:
        if dominio == 'generacion':
            from configuracion.acceso_datos.strategies_generacion import GeneracionStrategy as Strategy
        else:
            from configuracion.acceso_datos.strategies_consumo import ConsumoStrategy as Strategy
        
        strategy = Strategy()
        context = DatasetContext(strategy)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Verifique que el archivo de configuración del dataset existe.")
        return
    
    model_repo = JoblibModelRepository()
    imputer_repo = JoblibImputerRepository()
    ui = PipelineUI(context, model_repo, imputer_repo)
    ui.run()


if __name__ == "__main__":
    main()