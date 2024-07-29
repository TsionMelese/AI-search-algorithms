from random import randint, random
from timeit import repeat
from docx import Document
from matplotlib import pyplot as plt
from graph import Graph, Node
from search import GraphSearch

graph = Graph()
search_graph = GraphSearch(graph.adj_list)
search_graph.fileReader("location.txt")

Arad = graph.createNode('Arad', [('Sibiu', 140), ('Timisoara', 118), ('Zerind', 75)])
Sibiu = graph.createNode('Sibiu', [('Arad', 140), ('Oradea', 151), ('Fagaras', 99), ('Rimnicu Vilcea', 80)])
Zerind = graph.createNode('Zerind', [('Oradea', 71), ('Arad', 75)])
Timisoara = graph.createNode('Timisoara', [('Arad', 118), ('Lugoj', 111)])
Lugoj =  graph.createNode('Lugoj', [('Timisoara', 111), ('Mehadia', 70)])
Mehadia = graph.createNode('Mehadia', [('Lugoj', 70), ('Drobeta', 75)])
Drobeta = graph.createNode('Drobeta', [('Mehadia', 75), ('Craiova', 120)])
Craiova = graph.createNode('Craiova', [('Drobeta', 120), ('Rimnicu Vilcea', 146), ('Pitesti', 138)])
Rimnicu = graph.createNode('Rimnicu Vilcea', [('Sibiu', 80), ('Craiova', 146), ('Pitesti', 97)])
Oradea = graph.createNode('Oradea', [('Sibiu', 151), ('Zerind', 71)])
Fagaras = graph.createNode('Fagaras', [('Sibiu', 99), ('Bucharest', 211)])
Pitesti = graph.createNode('Pitesti', [('Rimnicu Vilcea', 97), ('Craiova', 138), ('Bucharest', 101)])
Bucharest = graph.createNode('Bucharest', [('Fagaras', 211), ('Pitesti', 101), ('Urziceni', 85), ('Giurgiu', 90)])
Urziceni = graph.createNode('Urziceni', [('Vaslui', 142), ('Hirsova', 98), ('Bucharest', 85)])
Hirsova = graph.createNode('Hirsova', [('Eforie', 86), ('Urziceni', 98)])
Eforie = graph.createNode('Eforie', [('Hirsova', 86)])
Vaslui = graph.createNode('Vaslui', [('Iasi', 92), ('Urziceni', 142)])
Iasi = graph.createNode('Iasi', [('Vaslui', 92), ('Neamt', 87)])
Neamt = graph.createNode('Neamt', [('Iasi', 87)])
Giurgiu = graph.createNode('Giurgiu', [('Bucharest', 90)])

# Add nodes to a list
nodes = [Eforie, Neamt, Iasi, Giurgiu, Arad, Sibiu, Zerind, Timisoara, Lugoj, Mehadia, Drobeta, Craiova, Rimnicu, Oradea, Fagaras, Pitesti, Bucharest, Urziceni, Hirsova, Vaslui, Iasi]

# You can use these nodes as needed in your program
for node in nodes:
    graph.addNode(node)

# Assuming nodes contains the list of nodes after creating them from the file
cities = [city for city in graph.adj_list.keys()]

random_cities = []
for i in range(10):
    while True:
        random_city = cities[randint(0, len(cities)-1)]
        if random_city not in random_cities:
            random_cities.append(random_city)
            break


# print(graph.astar('Bucharest', 'Arad'))


set_up = '''from __main__ import graph, search_graph, random_cities
from search import GraphSearch
from random import randint
from graph import Graph'''

bfs_code = '''
def test():
    for i in range(len(random_cities)):
        for j in range(i+1, len(random_cities)):
            search_graph.BFS(random_cities[i], random_cities[j])
'''

dfs_code = '''
def test():
    for i in range(len(random_cities)):
        for j in range(i+1, len(random_cities)):
            search_graph.DFS(random_cities[i], random_cities[j])
'''

astar_code = '''
def test():
    for i in range(len(random_cities)):
        for j in range(i+1, len(random_cities)):
            search_graph.astar(random_cities[i], random_cities[j])
'''

ucs_code = '''
def test():
    for i in range(len(random_cities)):
        for j in range(i+1, len(random_cities)):
            search_graph.UCS(random_cities[i], random_cities[j])
'''

iter_code = '''
def test():
    for i in range(len(random_cities)):
        for j in range(i+1, len(random_cities)):
            search_graph.iterativeDeepening(random_cities[i], random_cities[j], 5)
'''

greedy_code = '''
def test():
    for i in range(len(random_cities)):
        for j in range(i+1, len(random_cities)):
            search_graph.greedySearch(random_cities[i], random_cities[j])
'''

all_codes = [bfs_code, dfs_code, astar_code, ucs_code, iter_code, greedy_code]
alg_names = ['BFS', 'DFS', 'A*', 'UCS', 'Iterative Deepening', 'Greedy']

# Test each algorithm and record the results
results = []

all_paths = []
# for dfs
p = 0
for i in range(10):
    for j in range(i+1, 10):
        p += len(search_graph.DFS(random_cities[i], random_cities[j]))-1
all_paths.append(p)
# for A* and UCS
p = 0
for i in range(10):
    for j in range(i+1, 10):
        p += len(search_graph.astar(random_cities[i], random_cities[j]))-1
all_paths.append(p)
all_paths.append(p)
# for iterativeDeepening
p = 0
for i in range(10):
    for j in range(i+1, 10):
        p += len(search_graph.iterativeDeepening(random_cities[i], random_cities[j], 5))-1
all_paths.append(p)
# for BFS
p = 0
for i in range(10):
    for j in range(i+1, 10):
        p += len(search_graph.BFS(random_cities[i], random_cities[j]))-1
all_paths.append(p)
# for greedy
p = 0
for i in range(10):
    for j in range(i+1, 10):
        p += len(search_graph.greedySearch(random_cities[i], random_cities[j]))-1
all_paths.append(p)

for code in all_codes:
    result = repeat(setup=set_up, stmt=code, repeat=10, number=1)
    results.append(result)

# Record the average time taken for each algorithm
averages = [sum(result) / len(result) for result in results]

# Write results to a Word document
document = Document()

table = document.add_table(rows=1, cols=len(alg_names) + 1)
hdr_cells = table.rows[0].cells

hdr_cells[0].text = 'Algorithm'
for i, name in enumerate(alg_names):
    hdr_cells[i+1].text = name

for i in range(10):
    row_cells = table.add_row().cells
    row_cells[0].text = 'Trial ' + str(i+1)
    for j in range(len(averages)):
        row_cells[j+1].text = str(results[j][i])

row_cells = table.add_row().cells
row_cells[0].text = 'Average time'
for j in range(len(averages)):
    row_cells[j+1].text = str(averages[j])

row_cells = table.add_row().cells
row_cells[0].text = 'Path Length'
for i in range(6):
    row_cells[i+1].text = str(all_paths[i])


document.save('algorithm_performance.docx')
