import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Span:
    id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    parent_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        if self.end_time:
            return self.end_time - self.start_time
        return None

@dataclass
class Trace:
    id: str
    spans: List[Span] = field(default_factory=list)

class RequestTracer:
    def __init__(self):
        self.traces: Dict[str, Trace] = {}
        
    def start_trace(self) -> str:
        trace_id = str(uuid.uuid4())
        self.traces[trace_id] = Trace(id=trace_id)
        return trace_id
        
    def start_span(self, trace_id: str, name: str, parent_id: Optional[str] = None, tags: Dict[str, str] = None) -> str:
        if trace_id not in self.traces:
            self.traces[trace_id] = Trace(id=trace_id)
            
        span_id = str(uuid.uuid4())
        span = Span(
            id=span_id,
            name=name,
            start_time=time.time(),
            parent_id=parent_id,
            tags=tags or {}
        )
        self.traces[trace_id].spans.append(span)
        return span_id
        
    def end_span(self, trace_id: str, span_id: str):
        if trace_id in self.traces:
            for span in self.traces[trace_id].spans:
                if span.id == span_id:
                    span.end_time = time.time()
                    break
                    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        return self.traces.get(trace_id)
        
    def get_latency_breakdown(self, trace_id: str) -> Dict[str, float]:
        trace = self.get_trace(trace_id)
        if not trace:
            return {}
            
        breakdown = {}
        for span in trace.spans:
            if span.duration is not None:
                breakdown[span.name] = span.duration * 1000  # in ms
        return breakdown
