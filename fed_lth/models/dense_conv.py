import torch
from torch import Tensor
from typing import Tuple, Union
import datetime
import os
import ctypes
import uuid

from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple

import time
from math import ceil


nn = 32

class DenseConv2D(torch.nn.Module):
    
  def extract_dense(self, sparse_kernel):
    nrows = sparse_kernel.shape[0]
    ncols = sparse_kernel.shape[1]

    return [([i for i in range(nrows)], [i for i in range(ncols)])]

  def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t = 3, stride: _size_2_t  = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1, bias: bool = False):
    super(DenseConv2D, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.bias = bias

  
  def load(self, sparse_weight, bias):

    kernel_shape = sparse_weight.shape
    out_channels = kernel_shape[0]
    in_channels = kernel_shape[1]
    kernel_height = kernel_shape[2]
    kernel_width = kernel_shape[3]

    if (isinstance(self.kernel_size, Tuple)):
      self.kernel_height = self.kernel_size[0]
      self.kernel_width = self.kernel_size[1]
    else:
      self.kernel_height = self.kernel_size
      self.kernel_width = self.kernel_size


    #print(out_channels, self.out_channels, in_channels, self.in_channels, kernel_height, self.kernel_height,  kernel_width, self.kernel_width , kernel_height , kernel_width)

    assert(out_channels == self.out_channels and in_channels == self.in_channels and kernel_height == self.kernel_height and kernel_width == self.kernel_width and kernel_height % 2 == 1 and kernel_width % 2 == 1)


    # convert the sparse kernel weight into a sparse matrix and store in CSR format
    block_ptr = [0]
    kernel_ptr = []
    kernel_map = []
    kernel_offset = []
    kernel_value = []

    kernel_ptr_sparse = []
    kernel_map_sparse = []

    sparse_weight = sparse_weight.view(kernel_shape[0], kernel_shape[1] * kernel_shape[2] * kernel_shape[3])

    blocks = self.extract_dense(sparse_weight)

    for b in blocks:
      kernel_ptr.append(len(kernel_offset))
      for r in b[0]:
        kernel_offset.extend(b[1])
        kernel_value.extend(sparse_weight[r,b[1]].tolist())
        kernel_ptr.append(len(kernel_offset))
        kernel_map.append(r)
              
      kernel_map.append(-1)
      assert (len(kernel_ptr) == len(kernel_map))
      block_ptr.append(len(kernel_ptr))
    

    nrows = sparse_weight.shape[0]
    ncols = sparse_weight.shape[1]

    kernel_ptr_sparse.append(len(kernel_offset))

    #print(kernel_ptr_sparse)
    self.block_ptr = torch.IntTensor(block_ptr)
    self.kernel_ptr = torch.IntTensor(kernel_ptr)
    self.kernel_map = torch.IntTensor(kernel_map)
    self.kernel_offset = torch.IntTensor(kernel_offset)
    self.kernel_value = torch.FloatTensor(kernel_value)
    self.kernel_ptr_sparse = torch.IntTensor(kernel_ptr_sparse)
    self.kernel_map_sparse = torch.IntTensor(kernel_map_sparse) 

    self.filename = None


  def forward(self, input: Tensor) -> Tensor:  # input: HWCN
    input = input.transpose(0, 3).transpose(1, 2).transpose(0, 1)
    if not isinstance(self.dilation, Tuple):
      vertical_dilation = self.dilation
      horizontal_dilation = self.dilation
    else:
      vertical_dilation = self.dilation[0]
      horizontal_dilation = self.dilation[1]

    if not isinstance(self.stride, Tuple):
      vertical_stride = self.stride
      horizontal_stride = self.stride
    else:
      vertical_stride = self.stride[0]
      horizontal_stride = self.stride[1]

    if not isinstance(self.padding, Tuple):
      vertical_padding = self.padding
      horizontal_padding = self.padding
    else:
      vertical_padding = self.padding[0]
      horizontal_padding = self.padding[1]

    tmp_kernel_height = self.kernel_height + (self.kernel_height - 1) * (vertical_dilation -1)
    tmp_kernel_width = self.kernel_width + (self.kernel_width - 1) * (horizontal_dilation - 1)

    # get the input dimension, check if the dimension match with kernel dimension
    input_height = input.shape[0]
    input_width = input.shape[1]
    #print(input.shape)
    #print(self.in_channels)
    #print(self.out_channels)
    assert(input.shape[2] == self.in_channels)
    batch_size = input.shape[3]

    tmp = input_height - tmp_kernel_height + 2 * vertical_padding
    #assert(tmp % vertical_stride == 0)
    output_height = tmp // vertical_stride + 1
    tmp = input_width - tmp_kernel_width + 2 * horizontal_padding
    #assert(tmp % horizontal_stride == 0)
    output_width = tmp // horizontal_stride + 1

    output_channels = self.out_channels
    if self.filename == None:
      f = open('spmm_conv_n.cu', 'r')
      code_n = f.read()
      f.close()

      f = open('spmm_conv_sparse.cu', 'r')
      code_s = f.read()
      f.close()

      f = open('aspt_conv.cu', 'r')
      code_template = f.read()
      f.close()


      code_kernel = ''
      call_kernel = ''
      code_stream_decl = ''

      for i in range(len(self.block_ptr)-1):
        block_kernel_size = self.block_ptr[i+1] - self.block_ptr[i] - 1
        block_kernel_size = block_kernel_size.item()
        if block_kernel_size  < 1:
          continue

        code_stream_decl += f'cudaStream_t stream_{i};\n'


        if True:
          code_kernel += code_n.replace('_OWIDTH', str(output_width)).replace('_OHEIGHT', str(output_height)).replace('_OCHANNEL', str(output_channels)).replace('_STRIDE_HEIGHT', str(vertical_stride)).replace('_STRIDE_WIDTH', str(horizontal_stride)).replace('_PADDING_HEIGHT', str(vertical_padding)).replace('_PADDING_WIDTH', str(horizontal_padding)).replace('_KERNEL_HEIGHT', str(self.kernel_height)).replace('_KERNEL_WIDTH', str(self.kernel_width)).replace('_INPUT_HEIGHT', str(input_height)).replace('_INPUT_WIDTH', str(input_width)).replace('_DIALATION_HEIGHT', str(vertical_dilation)).replace('_DIALATION_WIDTH', str(horizontal_dilation)).replace('_INPUT_CHANNEL', str(self.in_channels)).replace('_BATCH_SIZE', str(batch_size)).replace('_NN', str(nn)).replace('_NKERNEL', str(block_kernel_size)).replace('_TOT_KERNEL', str(output_channels)).replace('_spmm_conv_n', f'_spmm_conv_{i}')
          call_kernel += f'cudaStreamCreate(&stream_{i});'
          call_kernel += f'\ndim3 nblocks_{i}({int((output_width*output_height*block_kernel_size/(4*nn)))}, {int((batch_size / 64))});\ndim3 nthreads_{i}(32, 4);\n_spmm_conv_{i}<<<nblocks_{i}, nthreads_{i}, 0, stream_{i}>>>(input_data, output_data, {self.block_ptr[i]}, {self.block_ptr[i+1]}, kernel_ptr, kernel_map, kernel_offset, kernel_data);\n'
        else:
          #assert (block_kernel_size < nn)
          code_kernel += code_n.replace('_OWIDTH', str(output_width)).replace('_OHEIGHT', str(output_height)).replace('_OCHANNEL', str(output_channels)).replace('_STRIDE_HEIGHT', str(vertical_stride)).replace('_STRIDE_WIDTH', str(horizontal_stride)).replace('_PADDING_HEIGHT', str(vertical_padding)).replace('_PADDING_WIDTH', str(horizontal_padding)).replace('_KERNEL_HEIGHT', str(self.kernel_height)).replace('_KERNEL_WIDTH', str(self.kernel_width)).replace('_INPUT_HEIGHT', str(input_height)).replace('_INPUT_WIDTH', str(input_width)).replace('_DIALATION_HEIGHT', str(vertical_dilation)).replace('_DIALATION_WIDTH', str(horizontal_dilation)).replace('_INPUT_CHANNEL', str(self.in_channels)).replace('_BATCH_SIZE', str(batch_size)).replace('_NN', str(block_kernel_size)).replace('_NKERNEL', str(block_kernel_size)).replace('_TOT_KERNEL', str(output_channels)).replace('_spmm_conv_n', f'_spmm_conv_{i}')
          call_kernel += f'cudaStreamCreate(&stream_{i});'
          call_kernel += f'\ndim3 nblocks_{i}({output_width*output_height//4}, {batch_size // 64});\ndim3 nthreads_{i}(32, 4);\n_spmm_conv_{i}<<<nblocks_{i}, nthreads_{i}, 0, stream_{i}>>>(input_data, output_data, {self.block_ptr[i]}, {self.block_ptr[i+1]}, kernel_ptr, kernel_map, kernel_offset, kernel_data);\n'



      if False:
        code_stream_decl += 'cudaStream_t stream_sparse;\n'
        sparse_kernel_size = len(self.kernel_ptr_sparse) - 1
        code_kernel += code_s.replace('_OWIDTH', str(output_width)).replace('_OHEIGHT', str(output_height)).replace('_OCHANNEL', str(output_channels)).replace('_STRIDE_HEIGHT', str(vertical_stride)).replace('_STRIDE_WIDTH', str(horizontal_stride)).replace('_PADDING_HEIGHT', str(vertical_padding)).replace('_PADDING_WIDTH', str(horizontal_padding)).replace('_KERNEL_HEIGHT', str(self.kernel_height)).replace('_KERNEL_WIDTH', str(self.kernel_width)).replace('_INPUT_HEIGHT', str(input_height)).replace('_INPUT_WIDTH', str(input_width)).replace('_DIALATION_HEIGHT', str(vertical_dilation)).replace('_DIALATION_WIDTH', str(horizontal_dilation)).replace('_INPUT_CHANNEL', str(self.in_channels)).replace('_BATCH_SIZE', str(batch_size)).replace('_NKERNEL', str(sparse_kernel_size)).replace('_TOT_KERNEL', str(output_channels))
        call_kernel += f'cudaStreamCreate(&stream_sparse);\ndim3 nblocks_sparse({output_width*output_height*sparse_kernel_size//2}, {batch_size // 64});\ndim3 nthreads_sparse(32, 2);\n_spmm_conv_sparse<<<nblocks_sparse, nthreads_sparse, 0, stream_sparse>>>(input_data, output_data, kernel_ptr_sparse, kernel_map_sparse, kernel_offset, kernel_data);\n'



      code = code_template.replace('_CODE_KERNEL', code_kernel).replace('_CODE_N', code_kernel).replace('_CALL_KERNEL', call_kernel).replace('_DECL_STREAM', code_stream_decl)

      timestamp = uuid.uuid1()
      self.filename = f'.tmp/tmp_{timestamp}'

      with open(self.filename+'.cu', 'w') as fw:
        fw.write(code)
      
      os.system(f'nvcc -gencode arch=compute_75,code=sm_75 -Xptxas "-v -dlcm=ca" -shared -Xcompiler=\"-fPIC\" -o {self.filename}.so {self.filename}.cu')

      self.kernel_ptr = self.kernel_ptr.cuda()
      self.kernel_map = self.kernel_map.cuda()
      self.kernel_offset = self.kernel_offset.cuda()
      self.kernel_value = self.kernel_value.cuda()
      self.kernel_ptr_sparse = self.kernel_ptr_sparse.cuda()
      self.kernel_map_sparse = self.kernel_map_sparse.cuda()

    output = torch.zeros(output_height, output_width, output_channels, batch_size).cuda()


    _libdir = os.path.dirname(os.path.realpath(__file__))
    _lib = ctypes.CDLL(self.filename+'.so')
    _lib.spmm_conv.restype = None
    _lib.spmm_conv.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

    #print(self.kernel_ptr_sparse)
    
    _lib.spmm_conv(ctypes.c_void_p(input.data_ptr()), ctypes.c_void_p(output.data_ptr()), ctypes.c_void_p(self.kernel_ptr.data_ptr()), ctypes.c_void_p(self.kernel_map.data_ptr()),  ctypes.c_void_p(self.kernel_offset.data_ptr()), ctypes.c_void_p(self.kernel_value.data_ptr()), ctypes.c_void_p(self.kernel_ptr_sparse.data_ptr()), ctypes.c_void_p(self.kernel_map_sparse.data_ptr()))
    #print(output.shape)
    return output.transpose(0, 1).transpose(0, 3).transpose(1, 2)




      






    


    