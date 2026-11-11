"""
Machine Learning Model Service - Loads and manages the solar production prediction model.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd


class MLModelService:
    """Service for loading and managing ML models for solar production prediction."""

    def __init__(self):
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.model_loaded: bool = False
        self.models_dir = Path(__file__).parent.parent.parent / "models"

    def load_model(self, model_name: str = "havana_v1") -> None:
        """
        Load the trained ML model and its metadata.

        Args:
            model_name: Name of the model to load (default: 'havana_v1')

        Raises:
            FileNotFoundError: If model files are not found
            RuntimeError: If model loading fails
        """
        try:
            # Construct file paths
            model_path = self.models_dir / f"solar_production_{model_name}.pkl"
            scaler_path = self.models_dir / f"scaler_{model_name}.pkl"
            metadata_path = self.models_dir / f"metadata_{model_name}.json"

            # Check if model file exists
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {model_path}. "
                    f"Please train the model by running the notebook at "
                    f"backend/notebooks/solar_production_prediction.ipynb"
                )

            # Load model
            print(f"Loading ML model from: {model_path}")
            self.model = joblib.load(model_path)

            # Load scaler if it exists (for linear models)
            if scaler_path.exists():
                print(f"Loading scaler from: {scaler_path}")
                self.scaler = joblib.load(scaler_path)

            # Load metadata
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"Model metadata loaded:")
                print(f"  Model: {self.metadata.get('model_name')}")
                test_rmse = self.metadata.get("test_rmse")
                test_r2 = self.metadata.get("test_r2")
                if test_rmse is not None:
                    print(f"  Test RMSE: {test_rmse:.4f}")
                if test_r2 is not None:
                    print(f"  Test R²: {test_r2:.4f}")
                print(f"  Features: {len(self.metadata.get('features', []))}")
                if self.metadata.get("reference_capacity_kw"):
                    print(f"  Reference capacity: {self.metadata['reference_capacity_kw']} kW")
            else:
                print(f"⚠️  Warning: Metadata file not found: {metadata_path}")
                self.metadata = {
                    "model_name": model_name,
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
                    "requires_scaling": False
                }

            self.model_loaded = True
            print(f"✓ ML model loaded successfully: {model_name}")

        except FileNotFoundError as e:
            print(f"❌ Error loading model: {e}")
            raise
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise RuntimeError(f"Failed to load ML model: {e}")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the loaded model.

        Args:
            features: DataFrame with features matching the model's expected input

        Returns:
            Array of predictions (solar production in kW)

        Raises:
            RuntimeError: If model is not loaded or prediction fails
        """
        if not self.model_loaded or self.model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() first or check if model file exists."
            )

        try:
            # Ensure features are in the correct order
            if self.metadata:
                expected_features = self.metadata.get("features", [])
                features = features[expected_features]

            # Apply scaling if required
            if self.metadata and self.metadata.get("requires_scaling") and self.scaler:
                features_scaled = self.scaler.transform(features)
                predictions = self.model.predict(features_scaled)
            else:
                predictions = self.model.predict(features)

            # Ensure non-negative predictions
            predictions = np.maximum(predictions, 0)

            return predictions

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """
        if not self.model_loaded:
            return {
                "loaded": False,
                "message": "Model not loaded"
            }

        return {
            "loaded": True,
            "model_name": self.metadata.get("model_name") if self.metadata else "unknown",
            "test_rmse": self.metadata.get("test_rmse") if self.metadata else None,
            "test_r2": self.metadata.get("test_r2") if self.metadata else None,
            "test_mae": self.metadata.get("test_mae") if self.metadata else None,
            "features": self.metadata.get("features") if self.metadata else [],
            "training_date": self.metadata.get("train_date") if self.metadata else None,
            "requires_scaling": self.metadata.get("requires_scaling") if self.metadata else False,
            "reference_capacity_kw": self.metadata.get("reference_capacity_kw") if self.metadata else None,
        }

    def get_reference_capacity_kw(self) -> Optional[float]:
        """
        Return the dataset reference capacity used during training, if available.
        """
        if not self.metadata:
            return None
        reference_capacity = self.metadata.get("reference_capacity_kw")
        try:
            return float(reference_capacity) if reference_capacity is not None else None
        except (TypeError, ValueError):
            return None

    def get_test_r2(self) -> Optional[float]:
        """
        Return the model's coefficient of determination (R²) on the test set.

        This is the *real* generalization metric measured during training and is
        the honest basis for a "confidence" figure shown to the user — not an
        ad-hoc formula.
        """
        if not self.metadata:
            return None
        r2 = self.metadata.get("test_r2")
        try:
            return float(r2) if r2 is not None else None
        except (TypeError, ValueError):
            return None


# Global singleton instance
ml_model_service = MLModelService()
