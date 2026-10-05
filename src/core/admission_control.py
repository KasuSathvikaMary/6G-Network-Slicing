import asyncio
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class AdmissionControl:
    def __init__(self, resource_pool=None):
        self.resource_pool = resource_pool
        self.policy = "STRICT"
        self.pending_queue: List[Dict[str, Any]] = []

    def set_policy(self, policy: str):
        if policy in ["STRICT", "BEST_EFFORT", "PREEMPTIVE"]:
            self.policy = policy
            
    async def check_admission(self, request: Dict[str, Any]) -> bool:
        if not self.resource_pool:
            return False
            
        required_compute = request.get('compute_cores', 4)
        required_mem = request.get('memory_gb', 8.0)
        priority = request.get('priority', 4)
        
        utilization = await self.resource_pool.get_utilization()
        
        # Simple feasibility check
        if self.policy == "STRICT":
            if utilization['compute_pct'] > 0.9 or utilization['memory_pct'] > 0.9:
                logger.warning("Rejecting slice request due to high utilization (STRICT policy)")
                self.pending_queue.append(request)
                return False
        elif self.policy == "PREEMPTIVE":
            if utilization['compute_pct'] > 0.95 and priority > 2:
                logger.warning("Rejecting low-priority slice request (PREEMPTIVE policy)")
                self.pending_queue.append(request)
                return False
                
        # Estimate feasibility
        is_feasible = True
        if self.resource_pool.available_compute < required_compute:
            is_feasible = False
        if self.resource_pool.available_memory < required_mem:
            is_feasible = False
            
        if is_feasible:
            logger.info("Slice request admitted.")
            return True
        else:
            logger.info("Slice request rejected. Queued for later.")
            self.pending_queue.append(request)
            return False

    async def process_queue(self):
        # Retry queued requests
        original_queue = list(self.pending_queue)
        self.pending_queue.clear()
        
        for req in original_queue:
            admitted = await self.check_admission(req)
            if not admitted:
                # Add it back to the queue (could implement priority queue in the future)
                pass
                
    async def estimate_resources(self, request: Dict[str, Any]) -> Dict[str, float]:
        return {
            "compute": request.get('compute_cores', 4) * 1.1,
            "memory": request.get('memory_gb', 8.0) * 1.1,
            "bandwidth": request.get('bandwidth_mbps', 1000.0)
        }
