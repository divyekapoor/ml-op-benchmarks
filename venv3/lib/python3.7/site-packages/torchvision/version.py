__version__ = '0.4.1.post2'
git_version = '7b6d9913f44e07609290a41ddb3c18b836d766a6'
from torchvision import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
