
from collections import defaultdict,deque

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx 

class Node:
    def __init__(self,value,lat,long,heruistic_cost=0):
        self.value = value
        self.heruistic_cost = heruistic_cost
        self.lat = float(lat)
        self.long = float(long) 
        
    
    def __lt__(self,other):
        return self.value < other.value

class Graph:
    def __init__(self):
        self.nodes = defaultdict(Node)
        self.edges = defaultdict(set)
        self.edge_cost = defaultdict(int)

    def insert(self,val,lat=0,long=0,heruistic_cost=0): 
        node = Node(val,lat,long,heruistic_cost)
        self.nodes[val] = node 
        return node

    def delete_node(self,node):
        for adjacent in self.edges[node]:
            self.edges[adjacent].remove(node)
            del self.edge_cost[tuple(sorted([node,adjacent]))]
        del self.nodes[node.value]
        del self.edges[node]
        

    def add_edge(self,val1,val2,edge_cost=None):
        node1 = self.nodes[val1]
        node2 = self.nodes[val2]

        if edge_cost == None:
            edge_cost = self.compute_edge_cost(node1,node2)

        self.edges[node1].add(node2)
        self.edges[node2].add(node1)
        self.edge_cost[tuple(sorted([node1,node2]))] = edge_cost

    def remove_edge(self,node1,node2):
        self.edges[node1].remove(node2)
        self.edges[node2].remove(node1)
        del self.edge_cost[tuple(sorted([node1,node2]))]
        
   

    def show(self):
        output = defaultdict(list)

        for val,node in self.nodes.items():
            for adjacent in self.edges[node]:
                cost = self.edge_cost[tuple(sorted([node,adjacent]))]
                output[node.value].append((adjacent.value,cost))
        
        return output

class SimpleNode:
    def __init__(self, val):
        self.val = val
        self.edges = []


class SimpleGraph:
    def __init__(self):
        self.nodes = []

    def create_node(self, val):
        newNode = SimpleNode(val)
        self.nodes.append(newNode)

    def add_edge(self, val1, val2):
        for node in self.nodes:
            if node.val == val1:
                node.edges.append(val2)
            if node.val == val2:
                node.edges.append(val1)

class Ranking:

    def __init__(self):
        self.cities = self.load_graph_from_file()
        self.graph = nx.Graph()
        self.graph.add_weighted_edges_from(self.cities)
        self.city = self.get_city_lst()
        self.city_dict = {self.city[i]: i for i in range(len(self.city))}

    def closeness_centrality(self):
        n, graph = self.get_city_adj_matrix()
        dist = self.floyd_warshall_closeness(n, graph)
        closeness = []
        city = self.get_city_lst()
        for i in range(n):
            total_distance = sum(dist[i])
            closeness_centrality = 1 / total_distance if total_distance != 0 else 0
            closeness.append([closeness_centrality, city[i]])

        closeness.sort()

        res = {}

        for city in closeness:
            res[city[1]] = city[0]
        return res

    def floyd_warshall_closeness(self, n, graph):
        dist = [[graph[i][j] for j in range(n)] for i in range(n)]

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
       
        return dist

    def degree(self):
        graph = SimpleGraph()
        cities = self.get_city_lst()
        for city in cities:
            graph.create_node(city)
        connection = self.load_graph_from_file()
        for city1, city2, _ in connection:
            graph.add_edge(city1, city2)

        res = {}

        for node in graph.nodes:
            res[node.val] = len(node.edges) / len(graph.nodes)
        return res

    def eigenvector(self):

        # create an adjacency matrix for the graph
        adjacency_matrix = np.array(self.get_city_adj_matrix_lite())

        # calculate the eigenvalues and eigenvectors of the adjacency matrix
        eigenvalues, eigenvectors = np.linalg.eig(adjacency_matrix)
        max_eigenvalue_index = np.argmax(eigenvalues)
        eigenvector = eigenvectors[:, max_eigenvalue_index]

        # normalize the eigenvector
        norm = np.linalg.norm(eigenvector)
        eigenvector_normalized = eigenvector / norm

        curCity = {}
        for city, val in self.city_dict.items():
            curCity[val] = city
        res = {}

        for i, ranking in enumerate(eigenvector_normalized):
            res[curCity[i]] = abs(ranking)

        return res

    def load_graph_from_file(self):
        cities = []
        edges = [('Arad', 'Sibiu', 140), ('Arad', 'Timisoara', 118), ('Arad', 'Zerind', 75),
         ('Sibiu', 'Oradea', 151), ('Sibiu', 'Fagaras', 99), ('Sibiu', 'Rimnicu Vilcea', 80),
         ('Zerind', 'Oradea', 71), ('Timisoara', 'Lugoj', 111), ('Lugoj', 'Mehadia', 70),
         ('Mehadia', 'Drobeta', 75), ('Drobeta', 'Craiova', 120), ('Craiova', 'Rimnicu Vilcea', 146),
         ('Craiova', 'Pitesti', 138), ('Rimnicu Vilcea', 'Pitesti', 97), ('Fagaras', 'Bucharest', 211),
         ('Pitesti', 'Bucharest', 101), ('Bucharest', 'Urziceni', 85), ('Bucharest', 'Giurgiu', 90),
         ('Urziceni', 'Vaslui', 142), ('Urziceni', 'Hirsova', 98), ('Hirsova', 'Eforie', 86),
         ('Vaslui', 'Iasi', 92), ('Iasi', 'Neamt', 87)]
        for city1, city2, cost in edges:
            cities.append((city1, city2, int(cost)))

        return cities

    def katz(self):

        # Create an adjacency matrix for a graph with 6 nodes and 6 edges
        A = np.array(self.get_city_adj_matrix_lite())

        # Set the damping factor and constant term
        alpha = 0.1
        beta = 1

        # Calculate Katz centrality using matrix multiplication
        n = len(self.get_city_adj_matrix_lite())
        I = np.identity(n)
        centrality = np.linalg.inv((I - (alpha*A)))
        centrality = beta * np.sum(centrality, axis=1)

        curCity = {}
        for city, val in self.city_dict.items():
            curCity[val] = city
        res = {}

        for i, ranking in enumerate(centrality):
            res[curCity[i]] = abs(ranking) - 1

        return res

    def pagerank_centrality(self):
        return nx.pagerank(self.graph)

    def get_city_lst(self):
        cities = self.load_graph_from_file()
        lst = set()
        for entry in cities:
            city1, city2, _ = entry
            lst.add(city1)
            lst.add(city2)
        return list(lst)

    def get_city_adj_matrix_lite(self):
        city = self.city
        city_with_cost = self.load_graph_from_file()
        city_dict = self.city_dict
        N = len(city)
        g = [[0]*N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i == j:
                    g[i][j] = 0
        for city1, city2, cost in city_with_cost:
            g[city_dict[city1]][city_dict[city2]] = 1
            g[city_dict[city2]][city_dict[city1]] = 1
        return g

    def get_city_adj_matrix(self):
        city = self.city
        city_with_cost = self.load_graph_from_file()
        city_dict = self.city_dict
        N = len(city)
        g = [[float("inf")]*N for _ in range(N)]

        for i in range(N):
            for j in range(N):
                if i == j:
                    g[i][j] = 0
        for city1, city2, cost in city_with_cost:
            g[city_dict[city1]][city_dict[city2]] = cost
            g[city_dict[city2]][city_dict[city1]] = cost
        
        return N, g

    def betweenness_centrality(self):
        graph = self.get_city_adj_matrix()[1]
        n = len(graph)
        betweenness = {node: 0 for node in range(n)}

        for s in range(n):
            stack = []
            pred = {node: [] for node in range(n)}
            dist = {node: -1 for node in range(n)}
            sigma = {node: 0 for node in range(n)}
            delta = {node: 0 for node in range(n)}
            dist[s] = 0
            sigma[s] = 1
            queue = [s]

            while queue:
                v = queue.pop(0)
                stack.append(v)
                for w in range(n):
                    if graph[v][w] != 0:
                        if dist[w] < 0:
                            queue.append(w)
                            dist[w] = dist[v] + 1
                        if dist[w] == dist[v] + 1:
                            sigma[w] += sigma[v]
                            pred[w].append(v)
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != s:
                    betweenness[w] += delta[w]

        return betweenness

    def pagerank_centrality(self):
        graph = self.get_city_adj_matrix()[1]
        tolerance = 1e-6
        max_iterations = 100
        damping_factor = 0.85
        n = len(graph)
        pagerank = {node: 1/n for node in range(n)}
        degree = [len(graph[node]) for node in range(n)]

        for i in range(max_iterations):
            new_pagerank = {node: (1 - damping_factor) /
                            n for node in range(n)}
            for j in range(n):
                for i in range(n):
                    if graph[i][j] != float("inf"):
                        new_pagerank[j] += damping_factor * \
                            pagerank[i] / degree[i]
            if all(abs(pagerank[node] - new_pagerank[node]) < tolerance for node in range(n)):
                break
            pagerank = new_pagerank
        curCity = {}
        for city, val in self.city_dict.items():
            curCity[val] = city

        res = {}
        for key in pagerank:
            res[curCity[key]] = pagerank[key]

        return res
    



class Experiment2:

    def __init__(self):
        self.ranking = Ranking()

    def plot_all_ranking(self):
        degree_ranking = self.ranking.degree()
        eigen = self.ranking.eigenvector()
        betweeness = self.ranking.betweenness_centrality()
        katz = self.ranking.katz()
        pagerank = self.ranking.pagerank_centrality()
        closeness = self.ranking.closeness_centrality()


        tableMapping = defaultdict(list)

        rankings = [degree_ranking, eigen, katz, pagerank, closeness]

        for ranking in rankings:
            for city in ranking:

                tableMapping[city].append(round(ranking[city], 5))

        table_data = [[k] + v for k, v in tableMapping.items()]

        fig, ax = plt.subplots()
        table = ax.table(cellText=table_data, colLabels=[
                         "Cities", "Degree", "EigenVector", "Katz", "Pagerank", "closeness"], loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        ax.axis("off")
        plt.show()
experiment = Experiment2()
experiment.plot_all_ranking()


rank = Ranking()
