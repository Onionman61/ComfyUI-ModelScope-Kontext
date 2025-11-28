# __init__.py

from .kontext_api import ModelScopeUniversalAPI

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "ModelScopeUniversalAPI": ModelScopeUniversalAPI
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelScopeUniversalAPI": "ModelScope Universal API"
}