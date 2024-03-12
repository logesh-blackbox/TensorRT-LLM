# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

class Mapping:
    def __init__(self, world_size, rank, gpus_per_node, tp_size=1, pp_size=1):
        if pp_size * tp_size != world_size:
            raise ValueError("world_size must equal to pp_size * tp_size")

        self.world_size = world_size
        self.rank = rank
        self.gpus_per_node = gpus_per_node
        self.tp_size = tp_size
        self.pp_size = pp_size

        self.pp_groups = [range(i, world_size, tp_size) for i in range(tp_size)]
        self.tp_groups = [range(i * tp_size, (i + 1) * tp_size) for i in range(pp_size)]

        self.pp_rank = self.rank // tp_size
        self.tp_rank = self.rank % tp_size

        self.tp_group = self.tp_groups[self.pp_rank]
        self.pp_group = self.pp_groups[self.tp_rank]

    def is_last_pp_rank(self):
        return self.pp_rank == self.pp_size - 1

    def is_first_pp_rank(self):
        return self.pp_rank == 0

    def has_pp(self):
        return self.pp_size > 1

    def prev_pp_rank(self):
        return (self.rank - self.tp_size) % self.world_size

    def next_pp_rank(self):
        return (self.rank + self.tp_size) % self.world_size
