import os
from typing import Dict
from ..system.log import Log


logger = Log(name="dataset:dataset_base")


def _singular_to_plural(name):
    if name.endswith('s'):
        return name + 'es'
    else:
        return name + 's'


class DatasetBase(object):
    def __init__(self, root, download=False, *args, metadata: Dict = {}, **kwargs):
        self.root = root
        self.metadata = metadata
        self.download = download
        self.__bootstrap__()

    def __load__(self):
        # To be implemented by subclasses to load data into the return items
        raise NotImplementedError("Subclasses must implement __load__ method")

    def __download__(self):
        # To be implemented by subclasses to download data if not present
        raise NotImplementedError(
            "Subclasses must implement __download__ method")

    @property
    def return_items(self):
        return iter(self.metadata.get("return_items", []))

    def _buffer_name(self, name):
        return f"_{name}{_singular_to_plural(name)}"

    def _getitem_name(self, name):
        return f"_get_{name}"

    def _unique_list_name(self, name):
        return f"_unique_{name}{_singular_to_plural(name)}"

    def get_buffer(self, name):
        return getattr(self, self._buffer_name(name), None)

    def set_buffer(self, name, value):
        setattr(self, self._buffer_name(name), value)

    def get_unique_list(self, name):
        return getattr(self, self._unique_list_name(name), None)

    def set_unique_list(self, name, value):
        setattr(self, self._unique_list_name(name), value)

    def buffer_getitem(self, name, index):
        buffer = self.get_buffer(name)
        if buffer is None:
            raise ValueError(f"Buffer for {name} not found")
        return lambda: buffer[index]

    def buffer_setitem(self, name, index, value):
        # For what?
        buffer = self.get_buffer(name)
        if buffer is None:
            raise ValueError(f"Buffer for {name} not found")
        buffer[index] = value

    def __bootstrap__(self):
        if self.download and not os.path.isdir(self.root):
            self.__download__()
        # Prepare the return items as specified in metadata
        for _name in self.return_items:
            self.set_buffer(_name, [])
        self.__load__()
        for _name in self.return_items:
            unique_items = list(set(self.get_buffer(_name)))
            self.set_unique_list(_name, unique_items)

    def __repr__(self):
        _fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        _fmt_str += f"    Number of datapoints: {self.__len__()}\n"
        _fmt_str += f"    Root Location: {self.root}\n"
        for _name in self.return_items:
            _unique_list = self.get_unique_list(_name)
            _len_unique = (
                len(_unique_list)
                if _unique_list is not None
                else "N/A"
            )
            _fmt_str += f"    {_singular_to_plural(_name)}: {_len_unique:>8d}\n"
        return _fmt_str

    def __len__(self):
        _itemlist = self.get_buffer(next(iter(self.return_items)))
        assert _itemlist is not None and len(_itemlist) > 0, \
            "Return items must non-empty. Did you forget to initialize them?"
        return len(_itemlist)

    def __getitem__(self, index):
        _ret = {}
        for _name in self.return_items:
            try:
                _ret[_name] = self.buffer_getitem(_name, index)()
            except Exception as exc:
                import traceback.format_exc as tbk
                logger.error(
                    '\n'
                    + f"Exception {_name} at index {index}:" + '\n'
                    + '\t' + exc + '\n'
                    + '\t\n'.join(tbk().splitlines())
                )
                _ret[_name] = None
        return _ret
