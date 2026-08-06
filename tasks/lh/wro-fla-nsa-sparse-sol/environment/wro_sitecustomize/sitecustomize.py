import sys
class _TLBlock:
    def find_module(self, name, path=None):
        return self if name=="tilelang" or name.startswith("tilelang.") else None
    def load_module(self, name):
        raise ImportError("tilelang disabled")
sys.meta_path.insert(0, _TLBlock())
