
file2 = open('dbpedia.taxonomy.txt', 'r')
file3 = open('dbpedia_new.taxnomy', 'w')
allchildren=[]
for line in file2:

    x=line[:-1].split("\t")
    a=x[0]
    b=x[1:]
    #newb=b
    print(set(b).intersection(allchildren))
    newb=set(b)-set(allchildren)
    allchildren=allchildren+b
    """for x in b:
                    if x in allchildren:
                        print(x)
                        newb.remove(x)
                    allchildren.append(x)"""
    file3.write(a+'\t'+('\t'.join(newb))+'\n')

file2.close()
file3.close()
 
