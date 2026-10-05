from dataclasses import dataclass
from typing import List

@dataclass
class NetworkNode:
    id: str
    type: str

@dataclass
class NetworkLink:
    src: str
    dst: str
    bandwidth: float

class TopologyGenerator:
    def __init__(self):
        self.nodes = []
        self.links = []

    def generate(self, topo_type: str, num_nodes: int):
        for i in range(num_nodes):
            self.add_node(f"n{i}", "base_station")

    def get_path(self, src: str, dst: str) -> List[str]:
        return [src, dst]

    def get_neighbors(self, node_id: str) -> List[str]:
        return []

    def add_node(self, node_id: str, node_type: str):
        self.nodes.append(NetworkNode(node_id, node_type))

    def remove_node(self, node_id: str):
        self.nodes = [n for n in self.nodes if n.id != node_id]
