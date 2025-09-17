#BASE_URL = "http://localhost:8100"
BASE_URL = "http://47.99.180.80/xgboost"
DOWNLOAD_URL = "./trained_models"
MCP_PORT = 8100
from pathlib import Path


def get_download_url(path:str):
    return f"{BASE_URL}/download/file/{Path(path).relative_to(DOWNLOAD_URL).as_posix()}"

def get_static_url(path:str):
    return f"{BASE_URL}/static/{Path(path).relative_to(DOWNLOAD_URL).as_posix()}"