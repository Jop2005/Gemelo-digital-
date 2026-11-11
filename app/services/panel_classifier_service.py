"""
Panel Dust Classification Service

Classifies solar panel images as clean or dusty using a pre-trained TensorFlow model.
The model is loaded once at application startup for optimal performance.
"""
import numpy as np
from pathlib import Path
from PIL import Image
import io
from typing import Optional, Dict
import tensorflow as tf
from tensorflow import keras


class PanelClassifierService:
    """
    Service for classifying solar panel cleanliness using ML model
    """

    def __init__(self):
        self.model: Optional[keras.Model] = None
        self.model_path = Path(__file__).parent.parent.parent / "notebooks" / "artifacts" / "best_model.keras"
        self.image_size = (224, 224)

    def load_model(self):
        """
        Load the pre-trained model from disk.
        Should be called once during application startup.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Please ensure the model has been trained and saved."
            )

        print(f"🤖 Loading panel classification model from {self.model_path}...")
        self.model = keras.models.load_model(self.model_path)
        print(f"✅ Model loaded successfully. Parameters: {self.model.count_params():,}")

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image for model inference.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Preprocessed image array ready for model prediction
        """
        # Load image from bytes
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Resize to model input size
        img_resized = img.resize(self.image_size)

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img_resized) / 255.0

        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)

        return img_batch

    def classify_panel(self, image_bytes: bytes) -> Dict[str, any]:
        """
        Classify a solar panel image as clean or dusty.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Dictionary with:
            - clasificacion: "limpio" or "sucio"
            - porcentaje_limpio: Percentage probability of being clean (0-100)
            - porcentaje_sucio: Percentage probability of being dusty (0-100)
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() during application startup."
            )

        # Preprocess image
        img_batch = self.preprocess_image(image_bytes)

        # Get prediction (returns value between 0 and 1)
        # Higher values indicate "dusty", lower values indicate "clean"
        prediction = self.model.predict(img_batch, verbose=0)[0][0]

        # Calculate percentages
        porcentaje_sucio = float(prediction * 100)
        porcentaje_limpio = float((1 - prediction) * 100)

        # Determine classification (threshold at 0.5)
        clasificacion = "sucio" if prediction > 0.5 else "limpio"

        return {
            "clasificacion": clasificacion,
            "porcentaje_limpio": round(porcentaje_limpio, 2),
            "porcentaje_sucio": round(porcentaje_sucio, 2)
        }


# Global instance (singleton)
panel_classifier_service = PanelClassifierService()
