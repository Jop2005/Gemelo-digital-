from typing import Dict, List, Any

class InterpretabilityModule:
    
    def __init__(self, models: Dict[str, Any], feature_names: List[str]):
        self.models = models
        self.feature_names = feature_names
    
    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        if model_name not in self.models:
            return {}
        model = self.models[model_name]
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            if len(self.feature_names) == len(importances):
                return dict(zip(self.feature_names, importances))
            return {f'feature_{i}': imp for i, imp in enumerate(importances)}
        return {}
    
    def compare_importances(self) -> Dict[str, Dict[str, float]]:
        return {name: self.get_feature_importance(name) for name in self.models}