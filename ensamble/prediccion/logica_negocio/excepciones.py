class PrediccionError(Exception):
    pass

class OrquestadorNoInicializado(PrediccionError):
    pass

class ModelosNoEncontrados(PrediccionError):
    pass