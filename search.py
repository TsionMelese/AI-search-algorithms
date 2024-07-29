from collections import deque
from queue import PriorityQueue
from math import sqrt

from collections import deque
from queue import PriorityQueue
from math import sqrt

class GraphSearch:
    def __init__(self, adj_list):
        self.adj_list = adj_list  # Initialize adjacency list
        self.location = {} 
    
    

    def BFS(self, start, goal):
        visted = {start}
        queue  = deque([(start, [start])])

        while queue:
            curr, path = queue.popleft()
            if curr == goal:
                return path

            for neighbour, _ in self.adj_list[curr]:

                if neighbour not in visted:
                    visted.add(neighbour)
                    queue.append((neighbour, path +  [neighbour]))

        return "Unreachable"

    def DFS(self, start, goal,):

        path = [start]
        visted = set()

        def dfs(start, goal):
            if start not in visted:
                visted.add(start)

                if start == goal:
                    return True
                
                for neighbour, cost in self.adj_list[start]:

                    path.append(neighbour)
                    isFound = dfs(neighbour, goal)

                    if isFound: return True

                    path.pop()
             
        if dfs(start, goal):
            return path

        return "Unreachable"


    def UCS(self, start, goal):
            queue = PriorityQueue()
            queue.put((0, start, []))

            visted = set()

            while not queue.empty():
                cost, node, path = queue.get()

                if goal == node:
                    # path = " => ".join(path + [node])
                    return f'total cost: {cost}   path: {path + [node]}'

                
                visted.add(node)

                for neighbour, edge_cost in self.adj_list[node]:
                    if neighbour in visted:
                        continue
                    
                    total_cost = cost + edge_cost
                    queue.put((total_cost, neighbour, path + [node]))

            return "No path to reach to {goal}".format(goal = goal)


    def iterativeDeepening(self,start, goal, maxDepth):

        def DLS(currNode, currDepth, path, visted ):

            if currNode == goal:
                path.append(currNode)
                return True
            
            if currDepth <= 0:
                return False
            
            visted.add(currNode)

            for neighbour, edge_cost in self.adj_list[currNode]:
                if neighbour in visted:
                    continue

                path.append(currNode)

                if DLS(neighbour, currDepth - 1, path, visted):
                    return True
                
                path.pop()


        for currDepth in range(maxDepth):
            visted, path = set(), []
            if DLS(start, currDepth, path, visted):
                return path
            
        return "Unreachable"


    def bidirectionalSearch(self, start, goal):
        forward_visited, backward_visited = {start}, {goal}

        forwardQueue, backwardQueue = PriorityQueue(), PriorityQueue()

        forwardQueue.put((0, start))  # (cost, node)
        backwardQueue.put((0, goal))  # (cost, node)

        from_start = {start: None}
        from_goal = {goal: None}

        while not forwardQueue.empty() and not backwardQueue.empty():

            if forwardQueue.qsize() <= backwardQueue.qsize():
                _, current = forwardQueue.get()

                if current in from_goal:
                    return self.path(from_start, neighbor, from_goal, start, goal)
                
              
                for neighbor, cost in self.adj_list[current]:
                    if neighbor not in forward_visited:

                        forward_visited.add(neighbor)
                        forwardQueue.put((cost, neighbor))

                        from_start[neighbor] = current

            else:
                _, current = backwardQueue.get()

                if current in from_start:
                    return self.path(from_start, neighbor, from_goal, start, goal)
               
                for neighbor, cost in self.adj_list[current]:
                    if neighbor not in backward_visited:

                        backward_visited.add(neighbor)
                        backwardQueue.put((cost, neighbor))
                        from_goal[neighbor] = current

        return "They aren't connected"


    def path(self, from_start, shared_node, from_goal, start, goal):
        path = [shared_node]
        node = shared_node

        while node != start:
            node = from_start[node]
            path.append(node)

        path = path[::-1]
        print(path)

        node = shared_node
        while node != goal:
            node = from_goal[node]
            path.append(node)

        return path


    def greedySearch(self, start, goal):

        queue = PriorityQueue()
        visted = set()

        heuristic_cost = self.heuristic(start, goal)
        queue.put((heuristic_cost, start))

        path = {start: None}

        while not queue.empty():
            _, node = queue.get()

            if node == goal:
                answerPath = []
                while node:
                    answerPath.append(node)
                    node = path[node]

                answerPath.reverse()
                return answerPath
            
            visted.add(node)

            for neighbour, _ in self.adj_list[node]:
                if neighbour in visted:
                    continue
                
                queue.put(( self.heuristic(neighbour, goal), neighbour))
                path[neighbour] = node
               

        return " Unreachable"
    
    def astar(self, start, goal):
        queue = PriorityQueue()
        visited = set()

        heuristic_cost = self.heuristic(start, goal)
        queue.put((0 + heuristic_cost, start))

        total_cost = {start: 0}
        path = {start: None}

        while not queue.empty():
            curr_cost, node = queue.get()

            if node == goal:
                answerPath = []
                while node:
                    answerPath.append(node)
                    node = path[node]

                answerPath.reverse()
                return answerPath

            visited.add(node)

            for neighbour, edge_cost in self.adj_list[node]:
                if neighbour in visited and curr_total_cost >= total_cost[neighbour]:
                    continue

                curr_total_cost = total_cost[node] + edge_cost

            # not evaluated yet but put in queue
                pending = [n for _, n in queue.queue]

                if neighbour not in pending:
                    queue.put((curr_total_cost + self.heuristic(neighbour, goal), neighbour))
                elif curr_total_cost >= total_cost[neighbour]:
                    continue

                path[neighbour] = node
                total_cost[neighbour] = curr_total_cost

        return "Unreachable"


    
    def heuristic(self, start, goal):
        if start not in self.location or goal not in self.location:
            raise ValueError("Start or goal location not found in the location dictionary.")
        return sqrt((self.location[start][0] - self.location[goal][0]) ** 2 + (self.location[start][1] - self.location[goal][1]) ** 2)

    def fileReader(self, fileName):
        with open(fileName, 'r') as file:
            for line in file:
                city, latitude, longtidue = line.strip().split("    ")
                self.location[city] = (float(latitude,), float(longtidue))