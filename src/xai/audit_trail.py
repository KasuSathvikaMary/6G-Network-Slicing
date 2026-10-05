import sqlite3
import json
import csv
from typing import Dict, List, Any, Optional
import time

class AuditTrail:
    """
    Maintains a record of all resource allocation decisions.
    """
    def __init__(self, db_path: str = "audit_trail.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    slice_id TEXT,
                    agent_id TEXT,
                    state_snapshot TEXT,
                    action_taken TEXT,
                    reward REAL,
                    top_features TEXT,
                    confidence REAL,
                    explanation TEXT
                )
            ''')
            conn.commit()

    def record_decision(self, slice_id: str, agent_id: str, state: List[float], 
                        action: List[float], reward: float, top_features: List[str], 
                        confidence: float, explanation: str):
        """
        Record a decision in the audit trail.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO decisions 
                (timestamp, slice_id, agent_id, state_snapshot, action_taken, reward, top_features, confidence, explanation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(), slice_id, agent_id, json.dumps(state), json.dumps(action),
                reward, json.dumps(top_features), confidence, explanation
            ))
            conn.commit()

    def query_decisions(self, slice_id: Optional[str] = None, 
                        start_time: Optional[float] = None, 
                        end_time: Optional[float] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query historical decisions based on filters.
        """
        query = "SELECT * FROM decisions WHERE 1=1"
        params = []
        
        if slice_id:
            query += " AND slice_id = ?"
            params.append(slice_id)
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        results = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            for row in rows:
                results.append({
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "slice_id": row["slice_id"],
                    "agent_id": row["agent_id"],
                    "state_snapshot": json.loads(row["state_snapshot"]),
                    "action_taken": json.loads(row["action_taken"]),
                    "reward": row["reward"],
                    "top_features": json.loads(row["top_features"]),
                    "confidence": row["confidence"],
                    "explanation": row["explanation"]
                })
        return results

    def export_report(self, filepath: str, format: str = 'json'):
        """
        Export all decisions to JSON or CSV.
        """
        decisions = self.query_decisions(limit=10000)
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(decisions, f, indent=2)
        elif format == 'csv':
            if not decisions:
                return
            keys = decisions[0].keys()
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for d in decisions:
                    row = dict(d)
                    row['state_snapshot'] = json.dumps(row['state_snapshot'])
                    row['action_taken'] = json.dumps(row['action_taken'])
                    row['top_features'] = json.dumps(row['top_features'])
                    writer.writerow(row)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_decision_stats(self) -> Dict[str, Any]:
        """
        Get aggregated statistics of decisions for compliance reporting.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(confidence), AVG(reward) FROM decisions")
            count, avg_conf, avg_reward = cursor.fetchone()
            
            cursor.execute("SELECT slice_id, COUNT(*) FROM decisions GROUP BY slice_id")
            slice_counts = dict(cursor.fetchall())
            
        return {
            "total_decisions": count or 0,
            "average_confidence": avg_conf or 0.0,
            "average_reward": avg_reward or 0.0,
            "decisions_per_slice": slice_counts
        }
