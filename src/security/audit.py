import sqlite3
import json
import time
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class AuditEvent:
    timestamp: float
    user: str
    action: str
    resource: str
    result: str
    details: str

class SecurityAuditor:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    user TEXT,
                    action TEXT,
                    resource TEXT,
                    result TEXT,
                    details TEXT
                )
            ''')
            conn.commit()
            
    def log_event(self, user: str, action: str, resource: str, result: str, details: dict = None):
        details_str = json.dumps(details or {})
        event = AuditEvent(time.time(), user, action, resource, result, details_str)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO audit_logs (timestamp, user, action, resource, result, details)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (event.timestamp, event.user, event.action, event.resource, event.result, event.details))
            conn.commit()
            
    def query_events(self, user: Optional[str] = None, action: Optional[str] = None, 
                     resource: Optional[str] = None, limit: int = 100) -> List[AuditEvent]:
        query = "SELECT timestamp, user, action, resource, result, details FROM audit_logs WHERE 1=1"
        params = []
        
        if user:
            query += " AND user = ?"
            params.append(user)
        if action:
            query += " AND action = ?"
            params.append(action)
        if resource:
            query += " AND resource = ?"
            params.append(resource)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
        return [AuditEvent(*row) for row in rows]
        
    def generate_report(self) -> dict:
        events = self.query_events(limit=1000)
        report = {
            "total_events": len(events),
            "failures": sum(1 for e in events if e.result.lower() == "failure"),
            "successes": sum(1 for e in events if e.result.lower() == "success"),
            "recent_events": [vars(e) for e in events[:10]]
        }
        return report
