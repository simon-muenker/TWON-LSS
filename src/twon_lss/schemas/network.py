import typing

import pydantic

import networkx

from .user import User


# TODO switch to igraph for improved performance
class Network(pydantic.BaseModel):
    """
    Represents the social network structure using NetworkX.

    The Network class models the connections and relationships between users
    in the social media simulation using a NetworkX graph structure.
    The class supports iteration over the network users.

    Attributes:
        graph (networkx.Graph): A NetworkX Graph object representing the network connections between users (default: empty graph).

    Example:
        >>> from src.twon_lss.schemas import Network
        ... import networkx as nx
        ... graph = nx.Graph()
        ... network = Network.from_graph(graph)
        ... neighbors = network.get_neighbors(user)
        ... for user in network:
        ...    print(user)
    """

    graph: networkx.Graph = networkx.Graph()

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def get_neighbors(self, user: User) -> typing.List[User]:
        return list(self.graph.neighbors(user))

    @classmethod
    def from_graph(cls, graph: networkx.Graph) -> "Network":
        return cls(structure=Network._relabel_to_users(graph))

    @staticmethod
    def _relabel_to_users(graph: networkx.Graph) -> networkx.Graph:
        return networkx.relabel_nodes(graph, mapping=lambda node_id: User(id=node_id))

    def __iter__(self):
        return iter(self.graph)

    def __next__(self):
        return next(self.graph)

    def __str__(self):
        return str(self.graph)
