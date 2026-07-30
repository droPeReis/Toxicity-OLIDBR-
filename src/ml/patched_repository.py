from typing import List, Union
from pathlib import Path
from huggingface_hub.utils import run_subprocess
import subprocess


def patched_files_to_be_staged(
    pattern: str = ".", folder: Union[str, Path, None] = None
) -> List[str]:
    """
    Returns a list of filenames that are to be staged.

    Args:
        pattern (`str` or `Path`):
            The pattern of filenames to check. Put `.` to get all files.
        folder (`str` or `Path`):
            The folder in which to run the command.

    Returns:
        `List[str]`: List of files that are to be staged.
    """
    try:
        # --exclude-standard
        p = run_subprocess(
            "git ls-files --exclude-standard -mo".split() + [pattern], folder
        )
        if len(p.stdout.strip()):
            files = p.stdout.strip().split("\n")
        else:
            files = []
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)

    return files
