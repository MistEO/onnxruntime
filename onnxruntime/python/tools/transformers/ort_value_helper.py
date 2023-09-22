import logging
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Optional

import numpy
import torch
from torch._C import _from_dlpack
from torch.utils.dlpack import to_dlpack
from torch._subclasses.fake_tensor import FakeTensor

from onnxruntime import InferenceSession, RunOptions
from onnxruntime.capi import _pybind_state as ORTC
from io_binding_helper import TypeHelper

logger = logging.getLogger(__name__)

def get_ort_device_type(device_type: str):
    if device_type == "cuda":
        return ORTC.OrtDevice.cuda()
    if device_type == "cpu":
        return ORTC.OrtDevice.cpu()
    raise ValueError("Unsupported device type: " + device_type)

def get_ort_devices(values: Tuple[torch.Tensor, ...]) -> Tuple[ORTC.OrtDevice, ...]:  # type: ignore
    assert all(value.device == values[0].device for value in values), "All values must be on the same device."

    def _device_id_or_zero(device_id: int) -> int:
        return device_id or 0

    devices: Tuple[ORTC.OrtDevice, ...] = tuple(  # type: ignore
        ORTC.OrtDevice(  # type: ignore
            get_ort_device_type(value.device.type),
            ORTC.OrtDevice.default_memory(),  # type: ignore
            _device_id_or_zero(value.device.index),
        )
        for value in values
    )
    return devices

def get_ortvalues_from_torch_tensors(
    tensors: Tuple[torch.Tensor, ...], devices: Tuple[ORTC.OrtDevice, ...]
) -> Tuple[torch.Tensor, ...]:
    ortvalues = ORTC.OrtValueVector()  # type: ignore
    ortvalues.reserve(len(tensors))
    dtypes = []
    shapes = []
    data_ptrs = []

    for tensor in tensors:
        dtypes.append(TypeHelper.torch_type_to_numpy_type(tensor.dtype))
        shapes.append(tensor.size())
        data_ptrs.append(tensor.data_ptr())
    ortvalues.push_back_batch(tensors, data_ptrs, dtypes, shapes, devices)
    return ortvalues

def to_real_tensor(tensor: FakeTensor) -> torch.Tensor:
    if tensor.is_sparse:
        raise ValueError("sparse tensor is not yet supported.")
    out = torch.empty(tensor.size(), dtype=tensor.dtype, device=tensor.device)
    return out


def ortvalues_to_torch_tensor(
    ortvalues: ORTC.OrtValueVector, device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, ...]:
    if len(ortvalues) == 0:
        return tuple()

    if device is not None and device.type == "ort":
        if not hasattr(ORTC, "to_aten_ort_device_tensor"):
            raise AttributeError("onnxruntime is missing to_aten_ort_device_tensor needed to support device == 'ort'.")
        return tuple(ORTC.to_aten_ort_device_tensor(ov) for ov in ortvalues)

    if not isinstance(ortvalues, ORTC.OrtValueVector):
        raise TypeError("ortvalues must be an instance of OrtValueVector not %r." % type(ortvalues))

    res: List[torch.Tensor] = ortvalues.to_dlpacks(_from_dlpack)
    bool_indices = ortvalues.bool_tensor_indices()
    if len(bool_indices):
        # DLPack structure does not know for sure if it stores boolean
        # or uint8. Method to_dlpacks cannot be used in that case.
        # Signature of *dl_packs* is `to_dlpacks(dlp, fct) -> list[torch.Tensor]`.
        # And fct is a function with signature `fct(dlp) -> torch.Tensor`.
        # Boolean tensors are converted into uint8 tensor with the DLPack protocol.
        # Therefore, the function `fct` does not know if the dlpack structure
        # is a boolean tensor or a uint8 tensor.
        # We could either consider another function as an input in
        # `to_dlpacks` or add an argument to `fct(dlp, ortvalue)`.
        # Second option makes it impossible to directly use `_from_dlpack` or
        # or `from_dlpack` from torch.
        # The best option would be to add boolean type in DLDataTypeCode.
        for i in range(0, len(bool_indices)):
            j = bool_indices[i]
            res[j] = res[j].to(torch.bool)

    return tuple(res)


def run_session_with_ortvaluevector(
    sess: InferenceSession,
    input_names: Tuple[str, ...],
    inputs: Tuple[torch.Tensor, ...],
    input_devices: Tuple[ORTC.OrtDevice, ...],  # type: ignore
    output_names: Tuple[str, ...],
    outputs: Tuple[torch.Tensor, ...],
    output_devices: Tuple[ORTC.OrtDevice, ...],  # type: ignore
    preallocate_output: bool,
) -> Tuple[torch.Tensor, ...]:

    inputs = tuple(a.contiguous() for a in inputs)

    ort_inputs = get_ortvalues_from_torch_tensors(inputs, input_devices)

    # preallocate output pytorch Tensors and use the buffers affined to the torch device for the output ortvalue.
    # Because the output ortvalue is not allocated and owned by ort, it does not need to convert the output ortvalue
    # to torch Tensor transferring the ownership.
    if preallocate_output:
        pth_outputs = tuple(map(lambda t: to_real_tensor(t) if isinstance(t, FakeTensor) else t, outputs))
        ort_outputs = get_ortvalues_from_torch_tensors(pth_outputs, output_devices)
    else:
        ort_outputs = ORTC.OrtValueVector()

    run_options = RunOptions()
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")
    sess.run_with_ortvaluevector(run_options, input_names, ort_inputs, output_names, ort_outputs, output_devices)

    if preallocate_output:
        return pth_outputs
    else:
        pth_outputs = ortvalues_to_torch_tensor(ort_outputs)  # type: ignore
        return pth_outputs








