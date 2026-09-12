"""
Module: Xanthommatin Graph State Registry
Description: Core non-linear memory continuity and structural topology engine.
Path: core/xanthommatin_registry.json
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Set, Any, Optional

class DeterministicInferenceCore:
    def __init__(self):
        self.state_telemetry: Dict[str, Any] = {}
        self.dissonance_log: List[str] = []

    def validate_and_register(self, node_id: str, state_data: Any) -> bool:
        self.state_telemetry[node_id] = state_data
        return True

    def broadcast_awareness(self) -> Dict[str, Any]:
        return self.state_telemetry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_telemetry": self.state_telemetry,
            "dissonance_log": self.dissonance_log
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeterministicInferenceCore":
        core = cls()
        core.state_telemetry = data.get("state_telemetry", {})
        core.dissonance_log = data.get("dissonance_log", [])
        return core


class CognitiveNode:
    def __init__(self, node_id: str, state_data: Any):
        self.node_id = node_id
        self.state_data = state_data
        self.coincident_planes: Set[str] = set()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "state_data": self.state_data,
            "coincident_planes": list(self.coincident_planes)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CognitiveNode":
        node = cls(data["node_id"], data["state_data"])
        node.coincident_planes = set(data.get("coincident_planes", []))
        return node


class OrthogonalPlane:
    def __init__(self, plane_id: str, vertex_ids: List[str]):
        self.plane_id = plane_id
        self.vertices: Set[str] = set(vertex_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plane_id": self.plane_id,
            "vertices": list(self.vertices)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrthogonalPlane":
        return cls(data["plane_id"], data.get("vertices", []))


class GraphStateRegistry:
    DEFAULT_PATH = "core/xanthommatin_registry.json"
    VERSION = "1.0.0"

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or self.DEFAULT_PATH
        self.di_core = DeterministicInferenceCore()
        self.nodes: Dict[str, CognitiveNode] = {}
        self.transit_planes: Dict[str, OrthogonalPlane] = {}

    def add_node(self, node_id: str, state_data: Any) -> CognitiveNode:
        self.di_core.validate_and_register(node_id, state_data)
        node = CognitiveNode(node_id, state_data)
        self.nodes[node_id] = node
        return node

    def add_transit_plane(self, plane_id: str, vertex_ids: List[str]):
        self.transit_planes[plane_id] = OrthogonalPlane(plane_id, vertex_ids)
        for v_id in vertex_ids:
            if v_id in self.nodes:
                self.nodes[v_id].coincident_planes.add(plane_id)

    def save_state(self, filepath: Optional[str] = None) -> None:
        """Serializes the registry with metadata header at the top of the JSON."""
        target_path = filepath or self.storage_path

        data = {
            "module": "Xanthommatin",
            "version": self.VERSION,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "di_core": self.di_core.to_dict(),
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "transit_planes": {pid: plane.to_dict() for pid, plane in self.transit_planes.items()}
        }

        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load_state(cls, filepath: Optional[str] = None) -> "GraphStateRegistry":
        """Restores the cognitive topology directly from the core path."""
        target_path = filepath or cls.DEFAULT_PATH
        
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Core registry state not found at: {target_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        registry = cls(storage_path=target_path)
        registry.di_core = DeterministicInferenceCore.from_dict(data.get("di_core", {}))
        
        registry.nodes = {
            nid: CognitiveNode.from_dict(ndata) 
            for nid, ndata in data.get("nodes", {}).items()
        }
        
        registry.transit_planes = {
            pid: OrthogonalPlane.from_dict(pdata) 
            for pid, pdata in data.get("transit_planes", {}).items()
        }

        return registry


if __name__ == "__main__":
    registry = GraphStateRegistry()
    registry.add_node("node_alpha", {"status": "active", "focus": "initialization"})
    registry.add_transit_plane("plane_1", ["node_alpha"])
    registry.save_state()
    print("Registry initialized and saved to core/xanthommatin_registry.json")