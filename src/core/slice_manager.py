import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import time
import uuid

logger = logging.getLogger(__name__)

class SliceType(Enum):
    eMBB = "eMBB"
    uRLLC = "uRLLC"
    mMTC = "mMTC"
    XR = "XR"

@dataclass
class SliceConfig:
    slice_type: SliceType
    bandwidth_mbps: float
    latency_ms: float
    compute_cores: int
    memory_gb: float
    priority: int
    sla_reliability_pct: float
    max_devices: int = 1000
    
@dataclass
class SliceInstance:
    slice_id: str
    config: SliceConfig
    state: str = "PENDING"
    created_at: float = field(default_factory=time.time)
    allocated_resources: Dict[str, Any] = field(default_factory=dict)
    
class SliceManager:
    def __init__(self, resource_pool=None, state_machine_factory=None):
        self.resource_pool = resource_pool
        self.state_machine_factory = state_machine_factory
        self.slices: Dict[str, SliceInstance] = {}
        self.templates: Dict[str, SliceConfig] = self._init_templates()
        
    def _init_templates(self) -> Dict[str, SliceConfig]:
        return {
            "XR": SliceConfig(SliceType.XR, 30000.0, 1.0, 16, 32.0, 1, 99.99),
            "uRLLC": SliceConfig(SliceType.uRLLC, 100.0, 0.1, 8, 16.0, 2, 99.999),
            "mMTC": SliceConfig(SliceType.mMTC, 1000.0, 50.0, 4, 8.0, 3, 99.9, max_devices=1000000),
            "eMBB": SliceConfig(SliceType.eMBB, 10000.0, 10.0, 8, 16.0, 4, 99.9)
        }

    async def create_slice(self, request: Dict[str, Any]) -> str:
        slice_id = str(uuid.uuid4())
        template_name = request.get('template', 'eMBB')
        config = self.templates.get(template_name)
        if not config:
            raise ValueError(f"Unknown template {template_name}")
            
        instance = SliceInstance(slice_id=slice_id, config=config)
        self.slices[slice_id] = instance
        logger.info(f"Created slice {slice_id} of type {template_name}")
        return slice_id

    async def provision_slice(self, slice_id: str):
        instance = self.slices.get(slice_id)
        if not instance:
            raise KeyError("Slice not found")
            
        instance.state = "PROVISIONING"
        if self.resource_pool:
            res = await self.resource_pool.allocate({
                "compute": instance.config.compute_cores,
                "memory": instance.config.memory_gb,
                "bandwidth": instance.config.bandwidth_mbps
            })
            instance.allocated_resources = res
        
        await asyncio.sleep(0.01)
        logger.info(f"Provisioned slice {slice_id}")

    async def activate_slice(self, slice_id: str):
        instance = self.slices.get(slice_id)
        if not instance:
            raise KeyError("Slice not found")
            
        instance.state = "ACTIVE"
        logger.info(f"Activated slice {slice_id}")

    async def monitor_slice(self, slice_id: str) -> Dict[str, Any]:
        instance = self.slices.get(slice_id)
        if not instance:
            return {}
        return {"slice_id": slice_id, "state": instance.state, "qos_met": True}

    async def optimize_slice(self, slice_id: str):
        instance = self.slices.get(slice_id)
        if not instance:
            return
        instance.state = "OPTIMIZING"
        await asyncio.sleep(0.01)
        instance.state = "ACTIVE"

    async def apply_optimization(self, slice_id: str, params: Dict[str, Any]):
        instance = self.slices.get(slice_id)
        if not instance:
            return
        logger.info(f"Applied optimization to {slice_id}: {params}")

    async def deactivate_slice(self, slice_id: str):
        instance = self.slices.get(slice_id)
        if instance:
            instance.state = "DEACTIVATING"
            await asyncio.sleep(0.01)
            instance.state = "TERMINATED"
            logger.info(f"Deactivated slice {slice_id}")

    async def terminate_slice(self, slice_id: str):
        instance = self.slices.get(slice_id)
        if instance:
            if self.resource_pool and instance.allocated_resources:
                await self.resource_pool.deallocate(instance.allocated_resources)
            del self.slices[slice_id]
            logger.info(f"Terminated slice {slice_id}")

    async def get_all_slices_status(self) -> List[Dict[str, Any]]:
        return [{"slice_id": sid, "state": s.state, "config": s.config} for sid, s in self.slices.items()]

    async def health_check(self) -> bool:
        return True

    async def reconfigure_slice(self, slice_id: str):
        instance = self.slices.get(slice_id)
        if instance:
            logger.info(f"Reconfigured slice {slice_id} to restore QoS")
            instance.state = "ACTIVE"
