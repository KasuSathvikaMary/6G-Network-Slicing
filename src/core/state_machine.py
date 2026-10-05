import logging
from typing import Dict, Any, Callable, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class SliceState(Enum):
    PENDING = "PENDING"
    PROVISIONING = "PROVISIONING"
    ACTIVE = "ACTIVE"
    OPTIMIZING = "OPTIMIZING"
    DEGRADED = "DEGRADED"
    MIGRATING = "MIGRATING"
    DEACTIVATING = "DEACTIVATING"
    TERMINATED = "TERMINATED"
    ERROR = "ERROR"

class StateMachine:
    def __init__(self, slice_id: str):
        self.slice_id = slice_id
        self.state = SliceState.PENDING
        self.history: List[Dict[str, Any]] = []
        self._transitions = self._define_transitions()
        self._callbacks: Dict[str, List[Callable]] = {}
        
    def _define_transitions(self) -> Dict[SliceState, List[SliceState]]:
        return {
            SliceState.PENDING: [SliceState.PROVISIONING, SliceState.ERROR],
            SliceState.PROVISIONING: [SliceState.ACTIVE, SliceState.ERROR],
            SliceState.ACTIVE: [SliceState.OPTIMIZING, SliceState.DEGRADED, SliceState.MIGRATING, SliceState.DEACTIVATING],
            SliceState.OPTIMIZING: [SliceState.ACTIVE, SliceState.ERROR],
            SliceState.DEGRADED: [SliceState.ACTIVE, SliceState.MIGRATING, SliceState.ERROR],
            SliceState.MIGRATING: [SliceState.ACTIVE, SliceState.ERROR],
            SliceState.DEACTIVATING: [SliceState.TERMINATED, SliceState.ERROR],
            SliceState.TERMINATED: [],
            SliceState.ERROR: [SliceState.PENDING, SliceState.TERMINATED]
        }
        
    def register_callback(self, event: str, callback: Callable):
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    async def _invoke_callbacks(self, event: str, *args, **kwargs):
        if event in self._callbacks:
            for cb in self._callbacks[event]:
                try:
                    import inspect
                    if inspect.iscoroutinefunction(cb):
                        await cb(*args, **kwargs)
                    else:
                        cb(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in state machine callback: {e}")

    async def transition(self, new_state: SliceState, reason: str = "") -> bool:
        if new_state in self._transitions[self.state]:
            old_state = self.state
            
            await self._invoke_callbacks(f"on_exit_{old_state.name}")
            
            self.state = new_state
            
            import time
            self.history.append({
                "from": old_state.name,
                "to": new_state.name,
                "reason": reason,
                "timestamp": time.time()
            })
            
            await self._invoke_callbacks("on_transition", old_state, new_state)
            await self._invoke_callbacks(f"on_enter_{new_state.name}")
            
            logger.info(f"Slice {self.slice_id} transitioned from {old_state.name} to {new_state.name}")
            return True
        else:
            logger.error(f"Invalid transition from {self.state.name} to {new_state.name} for slice {self.slice_id}")
            return False

    def get_state(self) -> SliceState:
        return self.state

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history
