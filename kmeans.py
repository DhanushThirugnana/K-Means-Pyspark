from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark import SparkContext

def func(x):
  leng=len(x)
  size=0
  while size<5 and size<leng:
      print("Cluster : " +str(x[size][0])+"\n"+x[size][1]+","+x[size][2]+","+x[size][3])
      size+=1

sc = SparkContext.getOrCreate()
file = sc.textFile("itemusermat")
file_data=file.map(lambda line:line.split(' ')[1:])
mid=file.map(lambda line:line.split(' ')[0])
d1=file_data
clusize = 10
iters=10
kmeansclus = KMeans.train(d1, clusize, iters)

d11=file.map(lambda x : (x.split(' ')[0],kmeansclus.predict(x.split(' ')[1:])))
file2=sc.textFile("movies.dat")
data2=file2.map(lambda line:line.split('::')).map(lambda x: (x[0],(x[1],x[2])))
datajoin=d11.join(data2)
datajoin=datajoin.map(lambda x: (x[1][0],x[0],x[1][1][0],x[1][1][1])).groupBy(lambda x: x[0]).map(lambda x : (x[0], list(x[1])))
out=datajoin.sortBy(lambda x:x[0]).map(lambda x: x[1])
out.foreach(func)
