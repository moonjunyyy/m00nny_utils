import os
from dataclasses import dataclass
from typing import Generator, TypeVar, List
from .threads import Thread
from ..util.formatting import bytes_to_human_readable
from ..system.log import Log

log = Log(name="PathTree")

_Node = TypeVar("_Node", bound="PathNode")
_Tree = TypeVar("_Tree", bound="PathTree")


class PathNode:
    @dataclass
    class Type:
        FILE = "f"
        LINK = "l"
        DIRECTORY = "d"
        OTHER = "o"

    def __init__(
        self,
        name: str,
        parent: _Node = None,
    ) -> None:
        self.name = name
        self.head = parent.head
        self.depth = parent.depth + 1
        self.depth_limit = parent.depth_limit
        self.parent = parent

        self.size = 0
        self.children = []
        self.parse()

    def __str__(self) -> str:
        _string = f"{self.depth * '  ' + '| '}{self.name}"
        _len_interval = 96 - len(_string)
        _string += (
            f"{' ' * _len_interval}" + f"{bytes_to_human_readable(size=self.size)}\n"
        )
        if self.type == PathNode.Type.DIRECTORY:
            for n, _child in enumerate(self.children):
                if n == 4:
                    _string += f"{self.depth * '  ' + '| '}...\n"
                    break
                _string += _child.__str__()
        return _string

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, PathNode):
            return self.name == other.name
        return False

    def __gt__(self, other) -> bool:
        return self.name > other.name  # Name sorting
        # return self.size > other.size # Size sorting

    def __lt__(self, other) -> bool:
        return self.name < other.name  # Name sorting
        # return self.size < other.size # Size sorting

    # Function for "in" operator
    def __contains__(self, key) -> bool:
        return key in self.children

    def is_empty(self) -> bool:
        if self.type != PathNode.Type.DIRECTORY:
            return False
        for _child in self.children:
            if _child.is_empty():
                _child.rm()
        return len(self.children) == 0

    @property
    def path(self) -> str:
        return os.path.join(self.parent.path, self.name) if self.parent else self.name

    @property
    def type(self) -> str:
        if os.path.isdir(s=self.path):
            return PathNode.Type.DIRECTORY
        if os.path.isfile(path=self.path):
            return PathNode.Type.FILE
        if os.path.islink(path=self.path):
            return PathNode.Type.LINK
        return PathNode.Type.OTHER

    @property
    def abs_path(self) -> str:
        return self.path

    @property
    def rel_path(self) -> str:
        return self.path.split(sep=self.head.path)[-1]

    def is_file(self) -> bool:
        return self.type == PathNode.Type.FILE

    def is_dir(self) -> bool:
        return self.type == PathNode.Type.DIRECTORY

    def is_link(self) -> bool:
        return self.type == PathNode.Type.LINK

    def get(self, name: str) -> _Node:
        for _child in self.children:
            if _child == name:
                return _child
        return None

    def remove(self, node: _Node) -> None:
        self.children.remove(node)

    def mkdir(self, name: str) -> _Node:
        os.mkdir(path=os.path.join(self.path, name))
        _node = PathNode(name=name, parent=self)
        self.children.append(_node)
        return _node

    def touch(self, name: str) -> _Node:
        open(file=os.path.join(self.path, name), mode="w").close()
        _node = PathNode(name=name, parent=self)
        self.children.append(_node)
        return _node

    def mv(self, dst: _Node | str) -> None:
        _th = Thread(target=self.__mv, args=(dst,), daemon=True)
        _th.start()
        _th.join()

    def __mv(self, dst: _Node | str) -> None:
        if isinstance(dst, str):
            dst = PathTree(path=dst, depth_limit=self.depth_limit)

        if self.name in dst.children:
            _subnode = dst.get(name=self.name)
            if self.type == PathNode.Type.DIRECTORY:
                log.debug(f"Moving {self.name} to {dst.name}...")
                _th_list = []
                for _child in self.children:
                    _th_list.append(
                        Thread(target=_child.__mv, args=(_subnode,), daemon=True)
                    )
                    _th_list[-1].start()
                for _th in _th_list:
                    _th.join()
                self.parent.remove(node=self.name)
                return
            else:
                log.warning(
                    f"{self.name} is already exists in {dst.name}. Overwriting..."
                )
                _subnode.rm()
        log.debug(f"Moving {self.name} to {dst.name}...")
        os.rename(src=self.path, dst=os.path.join(dst.path, self.name))
        self.parent.remove(node=self.name)
        self.parent = dst
        self.head = dst.head
        self.depth = dst.depth + 1
        dst.children.append(self)

    def rm(self) -> None:
        _th = Thread(target=self.__rm, daemon=True)
        _th.start()
        _th.join()

    def __rm(self) -> None:
        if self.type == PathNode.Type.FILE or self.type == PathNode.Type.LINK:
            try:
                os.remove(path=self.path)
            except Exception as e:
                pass
        elif self.type == PathNode.Type.DIRECTORY:
            if not self.is_empty():
                _th_list = []
                for _child in self.children:
                    _th_list.append(Thread(target=_child.__rm, daemon=True))
                    _th_list[-1].start()
                for _th in _th_list:
                    _th.join()
            try:
                os.rmdir(path=self.path)
            except Exception as e:
                pass
        if self.parent:
            self.parent.remove(node=self)

    def find(self, name: str) -> List[_Node]:
        _th = Thread(target=self.__find, args=(name,), daemon=True)
        _th.start()
        return _th.join()

    def __find(self, pattern: str) -> List[_Node]:
        _th_list = []
        _ret = []
        _candidates = []
        for _child in self.children:
            if _child.type == PathNode.Type.DIRECTORY:
                _th_list.append(
                    Thread(target=_child.__find, args=(pattern,), daemon=True)
                )
                _th_list[-1].start()
        for _th in _th_list:
            _ret += _th.join()
        for _child in self.children:
            if pattern == _child.name:
                _ret.append(_child)
            elif pattern in _child.name:
                _candidates.append(_child)
        if len(_ret) == 0:
            _ret += _candidates
        return _ret

    def __parse_dir(self) -> None:
        _lsdir = os.listdir(path=self.path)
        _th_list = []
        for _item in _lsdir:
            _node = PathNode.__new__(PathNode)
            _th = Thread(target=_node.__init__, args=(_item, self), daemon=True)
            _th.start()
            _th_list.append(_th)
            self.children.append(_node)
        for _th in _th_list:
            _th.join()

    def parse(self) -> None:
        if self.type == PathNode.Type.DIRECTORY:
            self.__parse_dir()
            self.size = sum([_child.size for _child in self.children])
        else:
            self.size = os.path.getsize(self.path)
        if self.depth == self.depth_limit:
            return

    def __iter__(self) -> Generator[_Node, None, None]:
        if self.type == PathNode.Type.DIRECTORY:
            for _child in self.children:
                yield from _child
        else:
            yield self

    def __getitem__(self, key) -> _Node:
        return self.find(pattern=key)


class PathTree(PathNode):
    def __init__(
        self,
        path: str,
        depth_limit: int = 5,
    ) -> None:
        self.name = os.path.abspath(path)
        self.depth = 0
        self.max_depth = 0
        self.depth_limit = depth_limit
        self.head = self
        self.parent = None

        self.size = 0
        self.children = []
        self.parse()

    @property
    def path(self) -> str:
        return self.name
