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

import pathlib as _pl
import subprocess as _sp
import typing as _tp


def run_command(command: _tp.Sequence[str], *, cwd=None, **kwargs) -> None:
    """
    Run a command in a specified directory.

    This function prints the command to be executed and then runs it using the `subprocess.check_call()`
    function. The command is run in the specified directory (`cwd`), and any additional keyword arguments
    are passed to `subprocess.check_call()`.

    Args:
    - command (Sequence[str]): The command to be executed, as a sequence of strings.
    - cwd (str, optional): The directory in which to run the command. Defaults to the current working directory.
    - **kwargs: Additional keyword arguments to be passed to `subprocess.check_call()`.

    Returns:
    None
    """
    print(f"Running: cd {str(cwd or _pl.Path.cwd())} && {' '.join(command)}")
    _sp.check_call(command, cwd=cwd, **kwargs)


# We can't use run_command() because robocopy (Robust Copy, rsync equivalent on Windows)
# for some reason uses nonzero return codes even on *successful* copies, so we need to check it manually.
# Also, robocopy only accepts dirs, not individual files, so we need a separate command for the
# single-file case.
def wincopy(source: str, dest: str, isdir: bool, cwd=None) -> None:
    """
    Copy a file or directory using robocopy on Windows.

    This function uses the `robocopy` command to copy a file or directory on Windows. If the source is a
    single file, it uses the `copy` command instead. The function prints the command to be executed and
    then runs it using the `subprocess.run()` function. The command is run in the specified directory (`cwd`),
    and the return code is checked to ensure that the copy was successful.

    Args:
    - source (str): The path to the source file or directory.
    - dest (str): The path to the destination file or directory.
    - isdir (bool): Whether the source is a directory. If True, `robocopy` is used; if False, `copy` is used.
    - cwd (str, optional): The directory in which to run the command. Defaults to the current working directory.

    Returns:
    None

    Raises:
    CalledProcessError: If the copy command returns a nonzero exit code.
    """
    if not isdir:  # Single-file copy
        run_command(["cmd", "/c", "copy",
                     str(_pl.Path(source)), f".\{dest}"],
                    cwd=cwd)
    else:  # Directory sync
        copy_cmd = ["robocopy", source, f"./{dest}", "/mir", "/e"]
        print(f"Running: cd {str(cwd or _pl.Path.cwd())} && {' '.join(copy_
