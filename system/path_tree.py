import os
from dataclasses import dataclass
from typing import Generator, TypeVar, List
from .task_pool import Task, Lock, RLock, Event, Condition
from ..util.formatting import bytes_to_human_readable
from ..system.log import Log

log = Log(name="PathTree")
_Node = TypeVar("_Node", bound="PathNode")
_Tree = TypeVar("_Tree", bound="PathTree")


class PathNode:
    @dataclass
    class Type:
        FILE = 'f'
        LINK = 'l'
        DIRECTORY = 'd'
        OTHER = 'o'

    def __init__(self,
                 name: str,
                 parent: _Node = None,
                 ) -> None:
        self.name = name
        self.parent = parent
        self.head = parent.head
        self.depth = parent.depth + 1
        self.depth_limit = parent.depth_limit
        self.size = 0
        self.children_lock = RLock()
        self.children = []
        self.parse()

    def __str__(self) -> str:
        _string = f"{self.depth * '  ' + '| '}{self.name}"
        _len_interval = 96 - len(_string)
        _string += (
            f"{' ' * _len_interval}" +
            f"{bytes_to_human_readable(size=self.size)}\n"
        )
        if self.type == PathNode.Type.DIRECTORY:
            for n, _child in enumerate(self.children):
                if n == 4:
                    _string += f"{self.depth * '  ' + '| '}...\n"
                    break
                _string += _child.__str__()
        return _string

    def __repr__(self) -> str: return self.__str__()

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
        with self.children_lock:
            _is_in = key in self.children
        return _is_in

    def __len__(self) -> int:
        with self.children_lock:
            _ret = len(self.children)
        return _ret

    def is_empty(self) -> bool:
        if self.type != PathNode.Type.DIRECTORY:
            return False
        for _child in self.children:
            if _child.is_empty():
                _child.rm()
        return len(self.children) == 0

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
    def path(self) -> str: return os.path.join(self.parent.path,
                                               self.name) if self.parent else self.name

    @property
    def abs_path(self) -> str: return self.path
    @property
    def rel_path(self) -> str: return self.path.split(sep=self.head.path)[-1]
    @property
    def is_file(self) -> bool: return self.type == PathNode.Type.FILE
    @property
    def is_dir(self) -> bool: return self.type == PathNode.Type.DIRECTORY
    @property
    def is_link(self) -> bool: return self.type == PathNode.Type.LINK

    def get(self, name: str) -> _Node:
        _found = None
        with self.children_lock:
            for _child in self.children:
                if _child == name:
                    _found = _child
                    break
        return _found

    def append(self, node: _Node) -> None:
        with self.children_lock:
            self.children.append(node)

    def remove(self, node: _Node) -> None:
        with self.children_lock:
            self.children.remove(node)

    def mkdir(self, name: str) -> _Node:
        os.mkdir(path=os.path.join(self.path, name))
        _node = PathNode(name=name, parent=self)
        with self.children_lock:
            self.children.append(_node)
        return _node

    def touch(self, name: str) -> _Node:
        open(file=os.path.join(self.path, name), mode="w").close()
        _node = PathNode(name=name, parent=self)
        with self.children_lock:
            self.children.append(_node)
        return _node

    def merge(self, dst: _Node | str) -> None:
        if isinstance(dst, str):
            dst = PathTree(path=dst, depth_limit=self.depth_limit)
        for _child in self.children:
            _child.mv(dst)
        self.rm()

    def mv(self, dst: _Node | str) -> None:
        if isinstance(dst, str):
            dst = PathTree(path=dst, depth_limit=self.depth_limit)
        if self.name in dst.children:
            _subnode = dst.get(self.name)
            if self.type == PathNode.Type.DIRECTORY:
                log.debug(f"Moving {self.name} to {dst.name}...")
                _th_list = []
                with self.children_lock:
                    for _child in self.children:
                        _th_list.append(
                            Task(target=_child.mv, args=(_subnode,), daemon=True))
                for _th in _th_list:
                    _th.start()
                for _th in _th_list:
                    _th.join()
                self.rm()
                return
            else:
                log.warning(
                    f"{self.name} is already exists in {
                        dst.name}. Overwriting..."
                )
                _subnode.rm()
        log.debug(f"Moving {self.name} to {dst.name}...")
        os.rename(src=self.path, dst=os.path.join(dst.path, self.name))
        src = self.parent
        self.parent = dst
        src.remove(self)
        self.head = dst.head
        self.depth = dst.depth + 1
        dst.append(self)

    def rm(self) -> None:
        if self.type == PathNode.Type.DIRECTORY:
            print(f"Delete {self.path}", flush=True)
            if not self.is_empty():
                _th_list = []
                with self.children_lock:
                    for _child in self.children:
                        _th_list.append(Task(target=_child.rm, daemon=True))
                for _th in _th_list:
                    _th.start()
                for _th in _th_list:
                    _th.join()
            try:
                os.rmdir(path=self.path)
                if self.parent:
                    self.parent.remove(node=self)
            except Exception as e:
                pass
        elif self.type == PathNode.Type.FILE or self.type == PathNode.Type.LINK:
            try:
                os.remove(path=self.path)
                if self.parent:
                    self.parent.remove(node=self)
            except Exception as e:
                pass

    def find(self, name: str) -> List[_Node]:
        _th_list = []
        _ret = []
        _candidates = []
        print(f"Finding {name} from {self.path}", flush=True)
        with self.children_lock:
            for _child in self.children:
                if _child.type == PathNode.Type.DIRECTORY:
                    _th_list.append(Task(target=_child.find,
                                    args=(pattern,), daemon=True))
        for _th in _th_list:
            _th.start()
        for _th in _th_list:
            _ret += _th.join()
        with self.children_lock:
            for _child in self.children:
                if pattern == _child.name:
                    _ret.append(_child)
                elif pattern in _child.name:
                    _candidates.append(_child)
        if len(_ret) == 0:
            _ret += _candidates
        return _ret

    def parse(self) -> None:
        if self.depth == self.depth_limit:
            return
        if self.type == PathNode.Type.DIRECTORY:
            _lsdir = os.listdir(path=self.path)
            _th_list = []
            for _item in _lsdir:
                _node = PathNode.__new__(PathNode)
                _th = Task(target=_node.__init__,
                           args=(_item, self), daemon=True)
                _th_list.append(_th)
                self.children.append(_node)
            for _th in _th_list:
                _th.start()
            for _th in _th_list:
                try:
                    _th.join()
                except Exception as e:
                    print(e, flush=True)
            with self.children_lock:
                self.size = sum([_child.size for _child in self.children])
        else:
            self.size = os.path.getsize(self.path)

    # def __iter__(self) -> Generator[_Node, None, None]:
    #     if self.type == PathNode.Type.DIRECTORY:
    #         with self.children_lock:
    #             for _child in self.children: yield from _child
    #     else: yield self

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
        self.children_lock = Lock()
        self.children = []
        self.parse()

    @property
    def path(self) -> str: return self.name
