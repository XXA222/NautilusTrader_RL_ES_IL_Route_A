"""Enforces TRAIN-only research *scope* on injected host providers.

This is a contract guard, NOT independent authorization. The host MUST reserve
ProjectResearchRegistry scope before any market read; never mint grants here.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol, TypeVar

T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True, slots=True)
class HostTrainGrant:
    """Host must create only after ProjectResearchRegistry accepted the access."""

    fold_id: str
    owner: str
    reservation_id: str
    source_identity: str
    split: str
    registered_before_read: bool

    def __post_init__(self) -> None:
        if self.split != "train" or self.registered_before_read is not True:
            raise PermissionError("Route-A collection requires a registered TRAIN-only source")
        if not all(isinstance(value, str) and value.strip() for value in
                   (self.fold_id, self.owner, self.reservation_id, self.source_identity)):
            raise PermissionError("missing host research scope identity")

    @classmethod
    def from_registered_read(cls, read: object, *, fold_id: str) -> "HostTrainGrant":
        """Bind the Route A contract to a host research read receipt.

        The receipt is produced only after the project registry reservation
        and scoped curated read have both succeeded.
        """
        if (getattr(read, "split", None) != "train"
                or getattr(read, "registered_before_read", None) is not True):
            raise PermissionError("host registered Train read receipt required")
        identity = getattr(read, "source_identity", None)
        if not isinstance(identity, dict):
            raise PermissionError("registered Train receipt has no source identity")
        marker = identity.get("marker_sha256")
        if not isinstance(marker, str) or re.fullmatch(r"[0-9a-fA-F]{64}", marker) is None:
            raise PermissionError("registered Train receipt has an invalid source marker")
        return cls(fold_id=fold_id, owner=getattr(read, "owner", ""),
                   reservation_id=getattr(read, "reservation_id", ""),
                   source_identity=marker, split="train",
                   registered_before_read=True)


class HostRegisteredReader(Protocol[T_co]):
    def read_registered_train(self, grant: HostTrainGrant) -> T_co: ...


def read_train_only(reader: HostRegisteredReader[T_co], grant: HostTrainGrant) -> T_co:
    """No generic read_file/read_parquet path is provided on purpose."""
    if not isinstance(grant, HostTrainGrant) or grant.split != "train":
        raise PermissionError("Train-only grant required")
    return reader.read_registered_train(grant)
