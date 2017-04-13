# Suppress GLOG output for python bindings
# unless explicitly requested in environment
import os
if 'GLOG_minloglevel' not in os.environ:
  # Hide INFO and WARNING, show ERROR and FATAL
  os.environ['GLOG_minloglevel'] = '2'
  _unset_glog_level = True
else:
  _unset_glog_level = False

from .pycaffe import SolverParameter, Net, SGDSolver, NesterovSolver, AdaGradSolver, RMSPropSolver, AdaDeltaSolver, AdamSolver
from ._caffe import set_mode_cpu, set_mode_gpu, set_device, Layer, set_devices, select_device, enumerate_devices, Layer, get_solver, get_solver_from_file, layer_type_list, set_random_seed
from ._caffe import __version__
from .proto.caffe_pb2 import TRAIN, TEST
from .classifier import Classifier
from .detector import Detector
from . import io
from .net_spec import layers, params, NetSpec, to_proto

if _unset_glog_level:
  del os.environ['GLOG_minloglevel']
