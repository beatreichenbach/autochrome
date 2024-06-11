from __future__ import annotations

import os


class File:
    """A File object as a snapshot in time.
    It is used to check if the file at the time of creation is the same as the file on
    disk now based on its modification time, not contents.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._hash = hash((path, os.path.getmtime(path)))

    def __add__(self, other: File) -> None:
        self.path += other

    def __eq__(self, other: File) -> bool:
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return f"File('{self.path}')"

    def __str__(self) -> str:
        return self.path
