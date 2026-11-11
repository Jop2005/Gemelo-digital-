import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

class InterpretabilityModule:
    
    def __init__(self, models: Dict[str, Any], feature_names: List[str]):
        self.models = models
        self.feature_names = feature_names
        self._shap_explainers: Dict[str, Any] = {}
    
    def explain_prediction(
        self, model_name: str, instance: pd.DataFrame,
        background_data: Optional[pd.DataFrame] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            import shap
        except ImportError:
            return {'error': 'SHAP no está instalado'}
        if model_name not in self.models:
            return {'error': f'Modelo {model_name} no encontrado'}
        instance_values = self._extract_instance_values(instance)
        try:
            explainer = self._get_or_create_explainer(model_name, background_data)
            shap_result = explainer.shap_values(instance_values)
            if hasattr(shap_result, 'values'):
                shap_values = shap_result.values.reshape(-1)
                base_value = float(shap_result.base_values.reshape(-1)[0])
            elif isinstance(shap_result, list):
                arr = shap_result[0] if len(shap_result) > 0 else shap_result
                shap_values = np.array(arr).reshape(-1)
                base_value = self._extract_base_value(explainer)
            else:
                shap_values = np.array(shap_result).reshape(-1)
                base_value = self._extract_base_value(explainer)
            prediction = base_value + np.sum(shap_values)
            return {
                'shap_values': [float(v) for v in shap_values],
                'base_value': base_value,
                'prediction': float(prediction),
                'features': self._extract_features_dict(instance),
                'contributions': self._build_contributions(shap_values)
            }
        except Exception as e:
            return {'error': f'Error al calcular SHAP: {str(e)}'}
    
    def _extract_instance_values(self, instance):
        if isinstance(instance, pd.DataFrame):
            if len(instance) != 1:
                raise ValueError('Solo una instancia a la vez')
            if all(f in instance.columns for f in self.feature_names):
                return instance[self.feature_names].values
            return instance.values
        return np.array(instance).reshape(1, -1)
    
    def _get_or_create_explainer(self, model_name, background_data=None):
        if model_name not in self._shap_explainers:
            import shap
            model = self.models[model_name]
            bg = self._prepare_background(background_data) if background_data is not None else None
            self._shap_explainers[model_name] = shap.TreeExplainer(model, bg) if bg is not None else shap.TreeExplainer(model)
        return self._shap_explainers[model_name]
    
    def _prepare_background(self, background_data):
        if isinstance(background_data, pd.DataFrame):
            return background_data[self.feature_names].values
        return background_data
    
    def _extract_base_value(self, explainer):
        bv = explainer.expected_value
        if hasattr(bv, 'values'):
            return float(bv.values.reshape(-1)[0])
        if isinstance(bv, np.ndarray):
            return float(bv.reshape(-1)[0])
        return float(bv)
    
    def _extract_features_dict(self, instance):
        if isinstance(instance, pd.DataFrame):
            d = instance.iloc[0].to_dict()
            return {f: d.get(f) for f in self.feature_names if f in d}
        return {self.feature_names[i]: float(instance.values[0, i]) if hasattr(instance, 'values') else float(instance[0][i])
                for i in range(min(len(self.feature_names), instance.shape[1] if hasattr(instance, 'shape') else len(instance)))}
    
    def _build_contributions(self, shap_values):
        return {self.feature_names[i]: float(shap_values[i])
                for i in range(min(len(self.feature_names), len(shap_values)))}