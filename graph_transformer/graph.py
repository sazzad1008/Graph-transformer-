"""Lightweight heterogeneous graph container."""

from collections import defaultdict
from typing import Any, Dict


class Graph:
    def __init__(self) -> None:
        super().__init__()
        self.node_forward = defaultdict(lambda: {})
        self.node_backward = defaultdict(lambda: [])

        self.node_features = defaultdict(lambda: [])
        self.edge_list = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: int)))
            )
        )
        self.times = set()

    def add_node(self, node: Dict[str, Any]) -> int:
        nfl = self.node_forward[node["type"]]
        if node["id"] not in nfl:
            self.node_backward[node["type"]] += [node]
            ser = len(nfl)
            nfl[node["id"]] = ser
            return ser
        return nfl[node["id"]]

    def add_edge(
        self,
        source_node: Dict[str, Any],
        target_node: Dict[str, Any],
        time: int | None = None,
        relation_type: str | None = None,
        directed: bool = True,
    ) -> None:
        edge = [self.add_node(source_node), self.add_node(target_node)]
        self.edge_list[target_node["type"]][source_node["type"]][relation_type][edge[1]][
            edge[0]
        ] = time
        if directed:
            self.edge_list[source_node["type"]][target_node["type"]][
                "rev_" + relation_type
            ][edge[0]][edge[1]] = time
        else:
            self.edge_list[source_node["type"]][target_node["type"]][relation_type][
                edge[0]
            ][edge[1]] = time
        self.times.add(time)

    def update_node(self, node: Dict[str, Any]) -> None:
        nbl = self.node_backward[node["type"]]
        ser = self.add_node(node)
        for k in node:
            if k not in nbl[ser]:
                nbl[ser][k] = node[k]

    def get_meta_graph(self) -> list[tuple[str, str, str]]:
        metas = []
        for target_type in self.edge_list:
            for source_type in self.edge_list[target_type]:
                for r_type in self.edge_list[target_type][source_type]:
                    metas += [(target_type, source_type, r_type)]
        return metas

    def get_types(self) -> list[str]:
        return list(self.node_features.keys())
