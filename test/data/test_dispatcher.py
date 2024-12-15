import pytest
from typing import Type
from GraphT5_3D.data.dispatcher import (
    DispatcherFactory,
    Dispatcher,
    DirectoryDispatcher,
)


@pytest.fixture
def mock_dispatcher() -> Type[Dispatcher]:
    @DispatcherFactory.register_dispatcher("mock")
    class MockDispatcher(Dispatcher):
        def load(self, **kwargs):
            return [1, 2, 3]

        def dispatch(self, idx: int):
            return self.src[idx]

    return MockDispatcher


def test_dispatcher_factory(mock_dispatcher):
    assert DispatcherFactory.dispatchers["mock"] == mock_dispatcher
    assert DispatcherFactory.get_dispatcher("mock") == mock_dispatcher


def test_DirectoryDispatcher(tmpdir):
    file1 = tmpdir.join("file1.txt")
    file1.write("file1")
    dispatcher = DirectoryDispatcher(from_dir=tmpdir)
    assert len(dispatcher) == 1
    assert dispatcher[0].pdb_path == str(file1)
