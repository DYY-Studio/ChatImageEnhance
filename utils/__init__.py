import sys
from pathlib import Path

def get_executable_dir():
    if hasattr(sys, 'frozen'):
        return Path(sys.executable).parent.resolve()
    else:
        return Path(__file__).parent.resolve()
    
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).parent.resolve()
    
    return str(base_path / relative_path)