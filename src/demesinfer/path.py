from jaxtyping import Scalar

Path = tuple[str | int, ...]
Value = Scalar | float


def get_path(params, path: Path) -> Value:
    for i in path:
        params = params[i]
    return float(params)


def set_path(params, path: Path, value: Value):
    for i in path[:-1]:
        params = params[i]
    params[path[-1]] = value


def bind(demo: dict, params: dict[Path, Value]):
    ret = dict(demo)
    for path, val in params.items():
        set_path(ret, path, val)
    return ret
