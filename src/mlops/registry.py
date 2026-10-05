import sqlite3
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelMetadata:
    name: str
    version: str
    status: str
    metrics: str

class ModelRegistry:
    def __init__(self, db_path=":memory:"):
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        self.conn.execute("CREATE TABLE IF NOT EXISTS models (name TEXT, version TEXT, status TEXT, metrics TEXT)")

    def register_model(self, name: str, version: str, metrics: str):
        self.conn.execute("INSERT INTO models VALUES (?, ?, ?, ?)", (name, version, "STAGING", metrics))

    def promote_model(self, name: str, version: str):
        self.conn.execute("UPDATE models SET status='PRODUCTION' WHERE name=? AND version=?", (name, version))

    def rollback(self, name: str, version: str):
        self.conn.execute("UPDATE models SET status='ARCHIVED' WHERE name=? AND version=?", (name, version))

    def get_production_model(self, name: str) -> Optional[ModelMetadata]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM models WHERE name=? AND status='PRODUCTION'", (name,))
        row = cur.fetchone()
        return ModelMetadata(*row) if row else None

    def list_models(self) -> List[ModelMetadata]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM models")
        return [ModelMetadata(*row) for row in cur.fetchall()]
