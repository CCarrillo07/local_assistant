import sqlite3
from contextlib import closing
from pathlib import Path

from memory.long_term_base import (
    LongTermMemoryStore,
    MemoryRecord
)

class SQLiteLongTermMemoryStore(
    LongTermMemoryStore
):

    def __init__(
        self,
        database_path: str | Path
    ):

        self.database_path = Path(
            database_path
        )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        self._initialize_database()

    def _initialize_database(self) -> None: 
        with closing(
            sqlite3.connect(
                self.database_path
            )
        ) as connection:
            with connection:
                connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories(
                    memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL UNIQUE
                )
                """
                )

    def remember(
        self,
        content: str    
    ) -> MemoryRecord:
        normalized_content = content.strip()

        if not normalized_content:
            raise ValueError(
                "Memory content cannot be blank"
            )

        with closing(
            sqlite3.connect(
                self.database_path
            )
        ) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO memories(content)
                    VALUES (?)
                    """,
                    (normalized_content,)
                )

                row = connection.execute(
                    """
                    SELECT memory_id, content
                    FROM memories
                    WHERE content = ?
                    """,
                    (normalized_content,)
                ).fetchone()

        return MemoryRecord(
            memory_id=row[0],
            content=row[1]
        )

    def list_memories(
            self
    ) -> list[MemoryRecord]:
        with closing(
            sqlite3.connect(
                self.database_path
            )
        ) as connection:
            rows= connection.execute(
                """
                SELECT memory_id, content
                FROM memories
                ORDER BY memory_id
                """
            ).fetchall()

        return [
            MemoryRecord(
                memory_id=row[0],
                content=row[1]
            )
            for row in rows
        ]

    def forget(
        self,
        memory_id: int
    ) -> bool:
        with closing(
            sqlite3.connect(
                self.database_path
            )
        ) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    DELETE FROM memories
                    WHERE memory_id = ?
                    """,
                    (memory_id,)
                )

                return cursor.rowcount > 0 
