import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import time
import uuid

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, slice_manager=None, resource_pool=None, admission_control=None, monitoring=None, drl_agent=None, energy_manager=None):
        self.orchestrator_id = str(uuid.uuid4())
        self.slice_manager = slice_manager
        self.resource_pool = resource_pool
        self.admission_control = admission_control
        self.monitoring = monitoring
        self.drl_agent = drl_agent
        self.energy_manager = energy_manager
        self.running = False
        self.event_queue = asyncio.Queue()
        self.status = "INITIALIZED"
        self._task_loop = None

    async def start(self):
        logger.info(f"Starting Orchestrator {self.orchestrator_id}")
        self.running = True
        self.status = "RUNNING"
        self._task_loop = asyncio.create_task(self._event_loop())
        
    async def stop(self):
        logger.info(f"Stopping Orchestrator {self.orchestrator_id}")
        self.running = False
        self.status = "STOPPED"
        if self._task_loop:
            self._task_loop.cancel()
            try:
                await self._task_loop
            except asyncio.CancelledError:
                pass
            
    async def _event_loop(self):
        while self.running:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self.handle_event(event)
                self.event_queue.task_done()
            except asyncio.TimeoutError:
                await self._perform_health_check()
            except Exception as e:
                logger.error(f"Error in event loop: {e}")

    async def handle_event(self, event: Dict[str, Any]):
        event_type = event.get('type')
        start_time = time.time()
        logger.debug(f"Handling event: {event_type}")
        
        if event_type == 'SLICE_REQUEST':
            await self.allocate_resources(event.get('request_data', {}))
        elif event_type == 'OPTIMIZATION_TRIGGER':
            await self.optimize_slices()
        elif event_type == 'DEGRADATION_DETECTED':
            await self._handle_degradation(event.get('slice_id'))
        else:
            logger.warning(f"Unknown event type: {event_type}")
            
        latency = (time.time() - start_time) * 1000
        logger.debug(f"Event {event_type} handled in {latency:.2f} ms")

    async def allocate_resources(self, request_data: Dict[str, Any]) -> bool:
        if not self.admission_control or not self.slice_manager:
            return False
            
        admitted = await self.admission_control.check_admission(request_data)
        if admitted:
            slice_id = await self.slice_manager.create_slice(request_data)
            await self.slice_manager.provision_slice(slice_id)
            await self.slice_manager.activate_slice(slice_id)
            return True
        return False

    async def optimize_slices(self):
        if not self.drl_agent or not self.slice_manager:
            return
            
        slices_status = await self.slice_manager.get_all_slices_status()
        optimization_actions = await self.drl_agent.compute_optimization(slices_status)
        for action in optimization_actions:
            await self.slice_manager.apply_optimization(action['slice_id'], action['params'])

    def get_status(self) -> Dict[str, Any]:
        return {
            "orchestrator_id": self.orchestrator_id,
            "status": self.status,
            "running": self.running,
            "components": {
                "slice_manager": self.slice_manager is not None,
                "resource_pool": self.resource_pool is not None,
                "admission_control": self.admission_control is not None
            }
        }
        
    async def _perform_health_check(self):
        if self.slice_manager:
            health = await self.slice_manager.health_check()
            if not health:
                logger.error("Slice manager health check failed, initiating self-healing")
                await self._heal_component('slice_manager')

    async def _heal_component(self, component_name: str):
        logger.info(f"Self-healing component: {component_name}")
        # Placeholder for healing logic, restarting components as needed
        if component_name == 'slice_manager' and self.slice_manager:
            await self.slice_manager.health_check() # retry or restart

    async def _handle_degradation(self, slice_id: str):
        if self.slice_manager:
            logger.warning(f"Slice {slice_id} degraded. Attempting recovery.")
            await self.slice_manager.reconfigure_slice(slice_id)
