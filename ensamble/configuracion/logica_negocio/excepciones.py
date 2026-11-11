class EvaluacionError(Exception):
    pass

class SegmentacionNoEncontrada(EvaluacionError):
    pass

class ModelosNoEncontrados(EvaluacionError):
    pass


class OptimizacionError(Exception):
    pass

class EvaluacionNoEncontrada(OptimizacionError):
    pass


class PrediccionError(Exception):
    pass

class OrquestadorNoInicializado(PrediccionError):
    pass

class ConfiguracionError(FileNotFoundError):
    pass