import random
import Graph

class RandomGraph(Graph.Graph):

    def add_random_edges(self, p=0.05):
        #Starts with an edgeless graph and adds edges at random, with a probability of p for any edge between two nodes to exist
        nodes = self.vertices()
        i = 0
        j = 0
        for v in nodes:
            for w in nodes:
                #Add edge between two nodes with probability p (if they're different nodes)
                if(j <= i):
                    j = j + 1
                    continue
                j = j + 1
                if(random.random() <= p):
                    self.add_edge(Graph.Edge(v,w))
            i = i + 1
            j = 0
