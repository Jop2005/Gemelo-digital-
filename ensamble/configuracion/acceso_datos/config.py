from pathlib import Path


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._inicializar()
        return cls._instance

    def _inicializar(self):
        self._root = Path(__file__).parent.parent.parent.resolve()
        self._carpeta_config = self._root / 'artifacts' / 'config' 
        self._carpeta_modelos = self._root / 'artifacts' / 'modelos'
        self._carpeta_imputers = self._root / 'artifacts' / 'imputers'
        self._carpeta_resultados = self._root / 'artifacts' / 'resultados'
        self._archivo_historial = self._root / 'artifacts' / 'historial_predicciones.csv'
        self._test_size = 0.2
        self._random_state = 42
        self._n_segments = 6
        self._crear_carpetas()

    @property
    def archivo_historial(self)-> Path:
        return self.archivo_historial

    def set_dominio(self, dominio: str):
        self._carpeta_config = self._root / 'artifacts' / 'config' / dominio
        self._carpeta_modelos = self._root / 'artifacts' / 'modelos' / dominio
        self._carpeta_imputers = self._root / 'artifacts' / 'imputers' / dominio
        self._carpeta_resultados = self._root / 'artifacts' / 'resultados' / dominio
        self._archivo_historial = self._root / 'artifacts' / f'historial_predicciones_{dominio}.csv'



    def _crear_carpetas(self):
        carpetas = [
            self._carpeta_config,
            self._carpeta_modelos,
            self._carpeta_imputers,
            self._carpeta_resultados,
        ]
        for carpeta in carpetas:
            carpeta.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def carpeta_config(self) -> Path:
        return self._carpeta_config

    @property
    def carpeta_modelos(self) -> Path:
        return self._carpeta_modelos

    @property
    def carpeta_imputers(self) -> Path:
        return self._carpeta_imputers

    @property
    def carpeta_resultados(self) -> Path:
        return self._carpeta_resultados

    @property
    def archivo_historial(self) -> Path:
        return self._archivo_historial

    @property
    def test_size(self) -> float:
        return self._test_size

    @property
    def random_state(self) -> int:
        return self._random_state

    @property
    def n_segments(self) -> int:
        return self._n_segments

    def set_carpeta_config(self, path):
        self._carpeta_config = Path(path)
        self._crear_carpetas()

    def set_carpeta_modelos(self, path):
        self._carpeta_modelos = Path(path)
        self._crear_carpetas()

    def set_test_size(self, value):
        self._test_size = value

    def set_n_segments(self, value):
        self._n_segments = value

def get_config():
    return Config()