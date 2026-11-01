from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import control as shared
shared.HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    getattr(shared,sys.argv[1])()
