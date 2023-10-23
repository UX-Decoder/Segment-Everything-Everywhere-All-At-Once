import json
import numpy as np
from typing import Dict


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JSONEncoder, self).default(obj)


def is_jsonable(x, json_encoder=None):
    try:
        json.dumps(x, cls=json_encoder)
        return True
    except Exception:
        return False


def filter_jsonable(data: Dict, json_encoder=None) -> Dict:
    return {k: v for k, v in data.items() if is_jsonable(k, json_encoder=json_encoder) and is_jsonable(v, json_encoder=json_encoder)}