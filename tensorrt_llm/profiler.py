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

import time

class Timer:
    """
    This class is used to track the time taken by different sections of the code.
    It starts a timer when the 'start' method is called with a specific tag,
    stops the timer when the 'stop' method is called with the same tag,
    and calculates the elapsed time. The 'elapsed_time_in_sec' method returns
    the total elapsed time for a specific tag. The 'reset' method clears all
    start and total elapsed times. The 'summary' method prints the total elapsed
    time for all tags.
    """

    def __init__(self):
        """
        Initializes the Timer object with empty dictionaries for start times and
        total elapsed times.
        """
        self._start_times = {}
        self._total_elapsed_times = {}

    def start(self, tag):
        """
        Starts the timer for the given tag.

        :param tag: The tag to start the timer for.
        """
        self._start_times[tag] = time.time()

    def stop(self, tag) -> float:
        """
        Stops the timer for the given tag, calculates the elapsed time, and adds
        it to the total elapsed time for that tag.

        :param tag: The tag to stop the timer for.
        :return: The elapsed time for the given tag.
        """
        elapsed_time = time.time() - self._start_times[tag]
        if tag not in self._total_elapsed_times:
            self._total_elapsed_times[tag] = 0
        self._total_elapsed_times[tag] += elapsed_time
        return elapsed_time

    def elapsed_time_in_sec(self, tag) -> float:
        """
        Returns the total elapsed time in seconds for the given tag.

        :param tag: The tag to get the elapsed time for.
        :return: The total elapsed time in seconds for the given tag.
        """
        if tag not in self._total_elapsed_times:
            return None
        return self._total_elapsed_times[tag]

    def reset(self):
        """
        Resets all start times and total elapsed times.
        """
        self._start_times.clear()
        self._total_elapsed_times.clear
