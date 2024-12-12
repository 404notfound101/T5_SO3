import os
from abc import ABC, abstractmethod
from typing import Dict, Type, Iterable
from glob import glob
from GraphT5_3D.data.protein import ProteinFile

class Dispatcher(ABC):
    def __init__(self, **kwargs):
        self.src = self.load(**kwargs)

    @abstractmethod
    def load(self, **kwargs) -> Iterable:
        pass

    @abstractmethod
    def dispatch(self, idx: int) -> ProteinFile:
        pass

    def __len__(self) -> int:
        return len(self.src)
    
    def __getitem__(self, idx) -> ProteinFile:
        return self.dispatch(idx)
    
class DispatcherFactory:
    dispatchers: Dict[str, Type[Dispatcher]] = {}

    @classmethod
    def register_dispatcher(cls, name: str):
        def decorator(dispatcher_cls: Type[Dispatcher]):
            if name in cls.dispatchers:
                raise ValueError(f"Dispatcher {name} already registered")
            cls.dispatchers[name] = dispatcher_cls
            return dispatcher_cls

        return decorator

    @classmethod
    def get_dispatcher(cls, name: str) -> Type[Dispatcher]:
        if name not in cls.dispatchers:
            raise ValueError(f"Dispatcher {name} not registered")
        return cls.dispatchers[name]
    
@DispatcherFactory.register_dispatcher("from_dir")
class DirectoryDispatcher(Dispatcher):
    def load(self, **kwargs) -> Iterable:
        if "from_dir" not in kwargs:
            raise ValueError("DirectoryDispatcher failed to initialize, no directory is provided")
        path_to_dir = kwargs["from_dir"]
        if not os.path.isdir(path_to_dir):
            raise ValueError(f"DirectoryDispatcher failed to initialize, {path_to_dir} does not exist")
        return glob(os.path.join(path_to_dir, "*"))

    def dispatch(self, idx: int) -> ProteinFile:
        return ProteinFile(pdb_path=self.src[idx], chains=["A"])
    

    


    