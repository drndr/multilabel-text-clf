import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt
 
g = nx.DiGraph()

file2 = open('dbpedia.taxonomy.txt', 'r')
for line in file2:
    x=line.split("\t")
    a=x[0]
    b=x[1][:-1]
    g.add_edge(a,b)
 
print(nx.is_tree(g))
nx.nx_agraph.write_dot(g,'test.dot')

# same layout using matplotlib with no labels
plt.title('draw_networkx')

pos =graphviz_layout(g, prog='dot')
nx.draw(g, pos, with_labels=False, arrows=False)
plt.savefig('nx_test.png')