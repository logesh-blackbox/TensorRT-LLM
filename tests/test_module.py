# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch

from tensorrt_llm.layers import GroupNorm
from tensorrt_llm.module import Module, ModuleList


class Module1(Module):
    """
    Module1 class with a custom name attribute and a forward method that registers a network output.
    """

    def __init__(self, name):
        """
        Initializes the Module1 class with a given name.

        :param name: str, the name of the module
        """
        super(Module1, self).__init__()
        self.name = name

    def forward(self):
        """
        Defines the forward pass of the Module1 class, registering a network output with the given index.

        :return: None
        """
        self.register_network_output('o1', 1)


class Module2(Module):
    """
    Module2 class that inherits from the Module class and has a Module1 instance and a custom name attribute.
    """

    def __init__(self):
        """
        Initializes the Module2 class with a custom name and a Module1 instance.
        """
        super(Module2, self).__init__()
        self.name = 'module2'
        self.m1 = Module1('m1')
        self.m2 = Module1('m2')

    def forward(self):
        """
        Defines the forward pass of the Module2 class, calling the forward pass of the Module1 instances and
        registering network outputs with the given indices.

        :return: None
        """
        self.m1.forward()
        self.m2.forward()
        self.register_network_output('o2', 2)
        self.register_network_output('o3', 3)


class Module3(Module):
    """
    Module3 class that inherits from the Module class and has a Module2 instance and a custom name attribute.
    """

    def __init__(self):
        """
        Initializes the Module3 class with a custom name and a Module2 instance.
        """
        super(Module3, self).__init__()
        self.name = 'module3'
        self.m1 = Module2()

    def forward(self):
        """
        Defines the forward pass of the Module3 class, calling the forward pass of the Module2 instance and
        registering a network output with the given index.

        :return: None
        """
        self.m1.forward()
        self.register_network_output('o4', 4)


class Module4(Module):
    """
    Module4 class that inherits from the Module class and has a ModuleList instance with two Module2 instances
    and a custom name attribute.
    """

    def __init__(self):
        """
        Initializes the Module4 class with a custom name and a ModuleList instance with two Module2 instances.
        """
        super(Module4, self).__init__()
        self.layers = ModuleList([Module2(), Module2()])
        self.name = 'module4'

    def forward(self):
        """
        Defines the forward pass of the Module4 class, calling the forward pass of each Module2 instance in the
        ModuleList and registering network outputs with the given indices.

        :return: None
        """
        for l in self.layers:
            l.forward()


class TestModule(unittest.TestCase):
    """
    TestModule class that inherits from the unittest.TestCase class and contains test methods for the Module
    classes.
    """

    def test_module(self):
        """
        Test method that creates an instance of Module3
