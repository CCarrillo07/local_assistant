from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass(frozen=True)
class MemoryRecord:
    memory_id: int
    content: str

class LongTermMemoryStore(ABC):

    @abstractmethod
    def remember(
        self,
        content: str 
    ) -> MemoryRecord:
        """Persist a user-approved fact."""
        raise NotImplementedError

    @abstractmethod
    def list_memories(
        self
    ) -> list[MemoryRecord]:
        """Return all persistent memories."""
        raise NotImplementedError

    @abstractmethod
    def forget(
        self,
        memory_id: int
    ) -> bool:
        """Remove one memory and report whether it existed."""
        raise NotImplementedError

                        