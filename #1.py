from geopy.distance import geodesic
import matplotlib.pyplot as plt

class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}

    def add_node(self, node_value):
        """Adds a new node to the graph."""
        self.nodes.add(node_value)

    def add_edge(self, source, destination):
        """Adds a directed edge from source to destination."""
        if source not in self.nodes:
            self.add_node(source)
        if destination not in self.nodes:
            self.add_node(destination)
        if source not in self.edges:
            self.edges[source] = []
        self.edges[source].append(destination)

    def remove_node(self, node_value):
        """Removes a node and its associated edges from the graph."""
        if node_value in self.nodes:
            self.nodes.remove(node_value)
            for neighbor in self.edges.get(node_value, []):  
                self.edges[neighbor].remove(node_value)
            del self.edges[node_value]  

    def remove_edge(self, source, destination):
        """Removes a directed edge from source to destination."""
        if source in self.edges and destination in self.edges[source]:
            self.edges[source].remove(destination)

    def load_from_file(self, filename):
        """Loads graph data from a text file."""
        with open(filename, 'r') as file:
            next(file)  
            for line in file:
                city, _, _ = line.strip().split(maxsplit=2)
                self.add_node(city)

    def __str__(self):
        return "[" + ", ".join([node for node in self.nodes]) + "]"


graph = Graph()

graph.load_from_file('cities.txt')

graph.add_edge('Oradea', 'Zerind')
graph.add_edge('Oradea', 'Sibiu')
graph.add_edge('Zerind', 'Arad')
graph.add_edge('Arad', 'Sibiu')
graph.add_edge('Arad', 'Timisoara')
graph.add_edge('Timisoara', 'Lugoj')
graph.add_edge('Lugoj', 'Mehadia')
graph.add_edge('Mehadia', 'Drobeta')
graph.add_edge('Drobeta', 'Craiova')
graph.add_edge('Craiova', 'RimnicuVilcea')
graph.add_edge('Craiova', 'Pitesti')
graph.add_edge('Sibiu', 'RimnicuVilcea')
graph.add_edge('Sibiu', 'Fagaras')
graph.add_edge('Fagaras', 'Bucharest')
graph.add_edge('Pitesti', 'Bucharest')
graph.add_edge('RimnicuVilcea', 'Pitesti')
graph.add_edge('Bucharest', 'Giurgiu')
graph.add_edge('Bucharest', 'Urziceni')
graph.add_edge('Urziceni', 'Vaslui')
graph.add_edge('Urziceni', 'Hirsova')
graph.add_edge('Hirsova', 'Eforie')
graph.add_edge('Vaslui', 'Iasi')
graph.add_edge('Iasi', 'Neamt')

positions = {}
with open('cities.txt', 'r') as file:
    next(file)  
    for line in file:
        city, latitude, longitude = line.strip().split()
        latitude = float(latitude)
        longitude = float(longitude)
        positions[city] = (longitude, latitude)  

plt.figure(figsize=(10, 8))

for node, (x, y) in positions.items():
    plt.plot(x, y, 'o', markersize=15, markerfacecolor='red', markeredgewidth=2, markeredgecolor='blue')
    plt.text(x, y + 0.1, node, ha='center', va='center', fontsize=12, weight='bold')  

for source, destinations in graph.edges.items():
    source_x, source_y = positions[source]
    for destination in destinations:
        dest_x, dest_y = positions[destination]
        plt.plot([source_x, dest_x], [source_y, dest_y], 'b-', alpha=0.7)

plt.title('Romania Cities Graph')
plt.show()
