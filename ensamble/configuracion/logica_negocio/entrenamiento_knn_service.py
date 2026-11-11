from sklearn.impute import KNNImputer


class EntrenamientoKNNService:
    """Servicio de lógica de negocio para entrenamiento del KNNImputer."""

    def __init__(self, context, imputer_repo):
        self._context = context
        self._imputer_repo = imputer_repo

    def ejecutar(self) -> KNNImputer:
        loader = self._context.get_loader()
        df = loader.obtener_features()

        knn = KNNImputer(n_neighbors=5)
        knn.fit(df)

        self._imputer_repo.guardar(knn, 'knn_imputer')

        return knn
