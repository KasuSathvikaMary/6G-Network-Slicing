import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional
import threading

logger = logging.getLogger(__name__)

@dataclass
class ResourceLimits:
    spectrum_mhz: float
    compute_cores: float
    memory_gb: float
    storage_gb: float
    energy_budget_w: float

class ResourcePool:
    def __init__(self, limits: ResourceLimits, oversub_ratio: float = 1.0):
        self.limits = limits
        self.oversub_ratio = oversub_ratio
        
        self.available_spectrum = limits.spectrum_mhz * oversub_ratio
        self.available_compute = limits.compute_cores * oversub_ratio
        self.available_memory = limits.memory_gb * oversub_ratio
        self.available_storage = limits.storage_gb * oversub_ratio
        self.available_energy = limits.energy_budget_w * oversub_ratio
        
        self.allocated = {
            "spectrum": 0.0,
            "compute": 0.0,
            "memory": 0.0,
            "storage": 0.0,
            "energy": 0.0
        }
        
        self.reservations = {}
        self._lock = threading.Lock()
        
    async def allocate(self, req: Dict[str, float]) -> Dict[str, Any]:
        with self._lock:
            comp_req = req.get("compute", 0.0)
            mem_req = req.get("memory", 0.0)
            bw_req = req.get("bandwidth", 0.0)
            
            if (self.available_compute >= comp_req and 
                self.available_memory >= mem_req and 
                self.available_spectrum >= bw_req):
                
                self.available_compute -= comp_req
                self.available_memory -= mem_req
                self.available_spectrum -= bw_req
                
                self.allocated["compute"] += comp_req
                self.allocated["memory"] += mem_req
                self.allocated["spectrum"] += bw_req
                
                return {"compute": comp_req, "memory": mem_req, "spectrum": bw_req}
            else:
                raise ValueError("Insufficient resources for allocation")

    async def deallocate(self, resources: Dict[str, float]):
        with self._lock:
            comp = resources.get("compute", 0.0)
            mem = resources.get("memory", 0.0)
            spec = resources.get("spectrum", 0.0)
            
            self.available_compute += comp
            self.available_memory += mem
            self.available_spectrum += spec
            
            self.allocated["compute"] -= comp
            self.allocated["memory"] -= mem
            self.allocated["spectrum"] -= spec

    async def reserve(self, req: Dict[str, float], timeout_s: float = 60.0) -> str:
        import uuid
        res_id = str(uuid.uuid4())
        allocation = await self.allocate(req)
        
        with self._lock:
            self.reservations[res_id] = allocation
        
        async def expire_reservation():
            await asyncio.sleep(timeout_s)
            with self._lock:
                if res_id in self.reservations:
                    logger.warning(f"Reservation {res_id} expired")
                    await self.deallocate(self.reservations[res_id])
                    del self.reservations[res_id]
                
        asyncio.create_task(expire_reservation())
        return res_id

    async def get_utilization(self) -> Dict[str, float]:
        with self._lock:
            comp_cap = self.limits.compute_cores * self.oversub_ratio
            mem_cap = self.limits.memory_gb * self.oversub_ratio
            spec_cap = self.limits.spectrum_mhz * self.oversub_ratio
            return {
                "compute_pct": (self.allocated["compute"] / comp_cap) if comp_cap > 0 else 0.0,
                "memory_pct": (self.allocated["memory"] / mem_cap) if mem_cap > 0 else 0.0,
                "spectrum_pct": (self.allocated["spectrum"] / spec_cap) if spec_cap > 0 else 0.0
            }
            
    async def defragment(self):
        logger.info("Defragmenting resource pool...")
        await asyncio.sleep(0.01)
        logger.info("Defragmentation complete")
