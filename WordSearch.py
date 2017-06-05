import Graph
import heapq
import time
from collections import deque
import matplotlib.pyplot as plt
import numpy

def aStar(g, source, goal, heur):
    #Use A* algorithm to complete word search
    numVisited = 0
    #Set up dictionary of visited nodes, previous node, and distances
    visited = {}
    distances = {}
    prev = {}
    seen = {}
    for key in g:
        visited[key] = False
        distances[key] = 1000000
        prev[key] = None
        seen[key] = False
    #Set up priority queue for algorithm
    q = []
    #Push source node onto pq
    heapq.heappush(q, (heur[source], source))
    distances[source] = 0
    seen[source] = True
    #Search for dest
    while(len(q)!=0 and not visited[goal]):
        #Pop minimum weight item of pq (weight uses heuristic)
        curNodeInfo = heapq.heappop(q)
        curNode = curNodeInfo[1]
        numVisited = numVisited+1
        #Search all adjacent nodes
        neighbours = g.out_vertices(curNode)
        for neighbour in neighbours:
            if(visited[neighbour] == False and seen[neighbour] == False):
                #Add them to pq if they're univisited and not already in the pq
                distToAdj = distances[curNode] + 1
                distances[neighbour] = distToAdj
                heapq.heappush(q, ((heur[neighbour]+distToAdj), neighbour))
                seen[neighbour] = True
                prev[neighbour] = curNode #Record their parent
        #Mark node as visited
        visited[curNode] = True

    #Determine the path to the goal node (start at goal and trace back parents to source)
    if(visited[goal]):
        path = []
        path.append(goal)
        parent = prev[goal]
        while(parent != None):
            path.append(parent)
            parent = prev[parent]
        #Reverse the order and return (since path is calculated backwards)
        pathProperOrder = []
        for i in range(len(path)-1, -1,-1):
            pathProperOrder.append(path[i])
        return (pathProperOrder, numVisited)

    else:
        #Goal can't be reached from the specified source
        return (None,len(visited))


def BFS(g, source, goal):
    #Use BFS to find solution to word chess
    numVisited = 0
    #Set up visited and previous dictionaries
    visited = {}
    prev = {}
    for key in g:
        visited[key] = False
        prev[key] = None
    #Set up queue
    q = deque([source])
    #Search graph
    while(len(q)!=0 and not visited[goal]):
        #Pop head of queue
        curNode = q.popleft()
        #Mark node as visited
        visited[curNode] = True
        numVisited = numVisited + 1
        #Search all adjacent nodes
        neighbours = g.out_vertices(curNode)
        for neighbour in neighbours:
            if(visited[neighbour] == False and not (neighbour in q)):
                #Add neighbour to q if it's unvisited, and not already in q
                q.append(neighbour)
                prev[neighbour] = curNode

    #Determine path from source to goal (start at goal and trace back parents to source)
    if(visited[goal]):
        path = []
        path.append(goal)
        parent = prev[goal]
        while(parent != None):
            path.append(parent)
            parent = prev[parent]
        #Invert order of path and return (since path is calculated backwards)
        pathProperOrder = []
        for i in range(len(path)-1, -1,-1):
            pathProperOrder.append(path[i])
        return (pathProperOrder, len(visited))
    else:
        return (None,len(visited))

def readWords(filename, wordLen):
    fin = open(filename)
    words = []
    for line in fin:
        line = line.rstrip()
        line = line.lower()
        if(len(line) == wordLen):
            words.append(line)
    return words

def wordDiff(w1, w2):
    diff = 0
    for i in range(len(w1)):
        if(w1[i] != w2[i]):
            diff = diff+1
    return diff

def constructGraph(words):
    #Given a list of words, constructs a graph out of them. Edge between every pair of words that are one letter apart
    g = Graph.Graph()
    for word in words:
        g.add_vertex(word)
    for i in range(len(words)):
        for j in range(i+1,len(words)):
            if(wordDiff(words[i], words[j]) == 1):
                e = Graph.Edge(words[i], words[j])
                g.add_edge(e)
    return g

def calcHeuristic(g, dest):
    #Calculates heuristics for A* algorithm
    heur = {}
    for key in g:
        heur[key] = wordDiff(key, dest)
    return heur

def wordSearch(source, dest, wordLen):
    #Read all words in from dictionary
    words = readWords('words', wordLen)

    #Construct graph out of words
    g = constructGraph(words)

    #Calculate shortest path using BFS
    start = time.time()
    (pathB, numNodesDiscoveredB) = BFS(g, source, dest)
    end = time.time()
    bfsTime = end-start

    #Calculate heuristics for A* algorithm
    heur = calcHeuristic(g, dest)

    #Calculate shortest path using A*
    start = time.time()
    (pathA, numNodesDiscoveredA) = aStar(g, source, dest, heur)
    end = time.time()
    aTime = end-start

    #Return paths, number of nodes visited, and time taken by each algorithm
    return (pathA, pathB, numNodesDiscoveredA, numNodesDiscoveredB, aTime, bfsTime)

source = raw_input("Enter a 4-letter word (the source word): ")
dest = raw_input("Enter a 4-letter word (the destination word): ")
sols = wordSearch(source, dest, 4)

#Plot results
names = ['A*', 'BFS']
numNodes = [sols[2], sols[3]]
fig = plt.figure()
y_pos = numpy.arange(len(names))
fig.suptitle('Number of Nodes Visited for A* Vs. BFS')
plt.bar(y_pos, numNodes, align='center')
plt.xticks(y_pos, names)
plt.ylabel('Number of Nodes Visited')
plt.xlabel('Search Algorithm')
plt.show()
print('Solutions:')
print('A*', sols[0])
print('BFS', sols[1])
