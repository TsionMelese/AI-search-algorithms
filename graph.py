from collections import defaultdict
from math import sqrt

class Node:
    def __init__(self, val, neighbours=None):
        self.val = val
        self.neighbours = neighbours if neighbours is not None else []

class Graph:
    def __init__(self):
        self.adj_list = defaultdict(list)
        self.location = {}

    def createNode(self, val, neighbours=None):
        return Node(val, neighbours)

    def addNode(self, node):
        self.adj_list[node.val] = node.neighbours

    def addNodes(self, nodes):
        for node in nodes:
            self.addNode(node)

    def removeNode(self, node):
        self.adj_list.pop(node)

    def addNeighbour(self, node, neighbour, weight=1):
        if (neighbour, weight) in self.adj_list[node]:
            return
        self.adj_list[node].append((neighbour, weight))
        self.adj_list[neighbour].append((node, weight))

    def addNeighbours(self, node, neighbours=None):
        if neighbours is None:
            neighbours = []
        self.adj_list[node].extend(neighbours)
        for neighbour, weight in neighbours:
            self.adj_list[neighbour].append((node, weight))

    def removeNeighbour(self, node, neighbour, weight):
        if node in self.adj_list:
            try:
                self.adj_list[node].remove((neighbour, weight))
                self.adj_list[neighbour].remove((node, weight))
            except ValueError:
                print("neighbour doesn't exist")

    def getConnections(self, node):
        return self.adj_list[node]

    def printGraph(self):
        print(self.adj_list)

    def isNodeExists(self, node):
        return node in self.adj_list

    def fileReader(self, fileName):
        with open(fileName, 'r') as file:
            for line in file:
                city, latitude, longitude = line.strip().split("    ")
                self.location[city] = (float(latitude), float(longitude))
    def heuristic(self, start, goal):
        if start not in self.location or goal not in self.location:
            raise ValueError("Start or goal location not found in the location dictionary.")
        return sqrt((self.location[start][0] - self.location[goal][0]) ** 2 + (self.location[start][1] - self.location[goal][1]) ** 2)