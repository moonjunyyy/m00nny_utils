import os
import sys
import pprint
import importlib.util
from .utils import add_library_paths, get_unique_project_id

# Generate from current project directory unique hashed identifier.
# First, get the hash of the current project main file.
# _hash = get_project_id()
# _module_path = os.path.dirname(__file__)
# Check if the shared library is already compiled.
# if not os.path.exists(os.path.join(_module_path, f"lib{_hash}")):
    # Compile the shared library
    # os.system(f"cd {_module_path} && OUTPUT_POSTFIX={_hash} bash {os.path.join(_module_path, 'build.sh')}")
# Load the dependent shared library
# lib_path = os.path.join(os.path.dirname(__file__), f"lib{_hash}")
lib_path = os.path.join(os.path.dirname(__file__), f"lib")
spec     = importlib.util.spec_from_file_location("kmeans", os.path.join(lib_path, "libkmeans.so"))
module   = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# import importlib.machinery
# lib_path = os.path.join(os.path.dirname(__file__), "lib")
# loader = importlib.machinery.ExtensionFileLoader("kmeans", os.path.join(lib_path, "libkmeans.so"))
# module = loader.load_module()
# pprint.pprint(dict(module.__dict__))

# class KMeans:
#     def __init__(self, n_clusters:int, max_iter:int=100, batchsize:int=128, mode:str='euclidean', init:str='kmeans++', seed:int=None) -> None:...
#     def fit_predict(self, X:torch.Tensor)->torch.Tensor:...
#     def fit(self, X:torch.Tensor):...
#     def predict(self, X:torch.Tensor)->torch.Tensor:...
#     def init_centroids(self)->None:...
#     def update_labels(self)->None:...
#     def update_centroids(self)->None:...
#     def compute_distance_matrix(self, X:torch.Tensor, Y:torch.Tensor)->torch.Tensor:...
#     def get_centroids(self) -> torch.Tensor:...

# def __bootstrap__():
#     global __bootstrap__, __loader__, __file__
#     import sys, pkg_resources, imp
#     lib_path = os.path.join(os.path.dirname(__file__), "lib")
#     __file__ = os.path.join(lib_path, "libkmeans.so")
#     # __file__ = pkg_resources.resource_filename(__name__,'kmeans.so')
#     __loader__ = None; del __bootstrap__, __loader__
#     imp.load_dynamic(__name__,__file__)
# __bootstrap__()

# import kmeans
# _path = "/".join(__file__.split("/")[:-1])
# if not os.path.exists(_path + "/lib"): os.makedirs(_path + "/lib")
# from torch.utils.cpp_extension import load
# kmeans = load(name='kmeans', sources=[_path + "/csrc/kmeans.cpp"], is_python_module=True, build_directory=_path + "/csrc/build", with_cuda=True);
# sys.modules['kmeans'] = kmeans;
# KMeans = kmeans.KMeans;