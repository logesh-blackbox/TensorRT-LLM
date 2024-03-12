import inspect
from copy import copy
from dataclasses import dataclass, field
from functools import wraps
from typing import (Any, Callable, ClassVar, Dict, List, Optional, Set, Tuple,
                    TypeVar)

import tensorrt as trt

from .logger import logger
from .network import Network


class Layer:
    '''
    Layer is a wrapper for TensorRT's ILayer with several python-friendly helper functions.
    '''

    def __init__(self, network: Network, trt_layer: trt.ILayer):
        self.network = network
        self.trt_layer = trt_layer

        assert isinstance(self.network, Network)
        assert isinstance(self.trt_layer, trt.ILayer)

    def get_inputs(self, *indices: int):
        '''
        Get the input tensors of the layer.

        Parameters:
            idx: the indices of the input tensor, will return all inputs if left empty

        Returns:
            List[Tensor]
        '''
        from .functional import Tensor
        indices = indices if indices else range(self.trt_layer.num_inputs)

        ret = []
        for i in indices:
            assert i < self.trt_layer.num_inputs, f"Invalid input index {i} for layer {self.trt_layer.name}"

            tensor = self.trt_layer.get_input(i)
            tensor = Tensor(trt_tensor=tensor,
                            network=self.network,
                            is_network_input=False)
            ret.append(tensor)
        return ret

    def get_outputs(self, *indices: int):
        '''
        Get the output tensor of the layer.

        Parameters:
            idx: the index of the output tensor

        Returns:
            List[Tensor]
        '''
        from .functional import Tensor

        indices = indices if indices else range(self.trt_layer.num_outputs)

        ret = []
        for i in indices:
            assert i < self.trt_layer.num_outputs, f"Invalid output index {i} for layer {self.trt_layer.name}"

            tensor = self.trt_layer.get_output(i)
            tensor = Tensor(trt_tensor=tensor,
                            network=self.network,
                            is_network_input=False)
            ret.append(tensor)
        return ret

    def is_removed(self):
        return self.network.is_removed_layer(self)

    def mark_as_removed(self):
        '''
        Mark the layer as removed, this will remove the layer from the network.
        '''
        # NOTE, since INetwork python API doesn't provide a way to remove a layer, we actually mark the layer as removed in the network.
        self.network.mark_removed_layer(self)

        # remove the FLayerInfo if exists
        FLayerInfoMemo.instance().remove(self.name)

    def __eq__(self, other: "Layer") -> bool:
        if isinstance(other, Layer):
            return self.trt_layer == other.trt_layer
        if isinstance(other, trt.tensorrt.ILayer):
            return self.trt_layer == other
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.trt_layer, name)


@dataclass
class _Pattern:
    name: str
    # args helps to pass in/out some information
    args: Dict[str, Any] = field(default_factory=dict, init=False)

    def log_info(self, msg: str):
        logger.info(f"Pattern {self.name}: {msg}")

    def log_error(self, msg: str):
        logger.error(f"Pattern {self.name}: {msg}")

    def log_warn(self, msg: str):
        logger.warning(f"Pattern {self.name}: {msg}")


class PatternRewriter(_Pattern):
    '''
    A pattern rewriter is a class that can match a pattern in the graph and rewrite the matched pattern.

    There are two ways to implement a pattern rewriter, either override match() and rewrite() separately, or override match_and_rewrite().
    '''

    def __init__(self,
                 name: str,
                 root_layer: Optional[Set[trt.LayerType]] = None,
                 seperate_match_rewrite=False):
        '''
        Parameters:
            name: the name of the rewrite pattern
            root_layer: the root layer types to start the pattern matching, if not provided, the pattern will traverse all the layers in the graph.
            seperate_match_rewrite: if set to True, the pattern should override match() and rewrite() separately, otherwise, the pattern should override match_and_rewrite()
        '''
        super().__init__(name)
        self.root_layer = root_layer
        self._seperate_match_rewrite = seperate_match_rewrite

    def match(self, layer: Layer) -> bool:
        raise NotImplementedError()

    def rewrite(self, layer: Layer) -> None:
        raise NotImplementedError()

    def match_and_rewrite(self, layer: Layer) -> bool:
        raise NotImplementedError()


class PatternAnalyzer(_Pattern):

    def __init__(self, name: str,
                 root_layer: Optional[Set[trt.LayerType]]) -> None:
        super().__init__(name)
        self.root_layer = root_layer

    def match(self, layer: Layer) -> bool:
        raise NotImplementedError()

    def analyze(self, subgraph: List[Layer]) -> None:
        raise NotImplementedError()


class _PatternManager:
    PatternType = TypeVar('PatternType')

    def __init__(self):
        # records of (benefit, pattern, id)
        self.patterns: Dict[str, Tuple[int, _PatternManager.
