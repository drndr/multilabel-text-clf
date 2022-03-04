
# Writing to file
label = {}
a_file = open("labels.txt")
for line in a_file:
    key, value = line.split('\t')
    label[key] = value[:-1]

# Opening file
file2 = open('label_hierarchy.txt', 'r')
file3 = open('dbpedia.taxnomy','w')
r=0
children=[]
alla=[]
allb=[]
# Using for loop
print("Using for loop")
for line in file2:
    x=line.split("\t")
    a=x[0]
    b=x[1][:-1]
    alla.append(a)
    allb.append(b)
    if r==int(a):
    	children.append(label[b])
    else:
    	file3.write(label[str(r)]+'\t'+('\t'.join(children))+'\n')
    	r=int(a)
    	children=[]
    	children.append(label[b])

print(set(alla).difference(allb))
# Closing files
file2.close()
file3.close()