import numpy as np
import random

import matplotlib.pyplot as plt
 
def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
	return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
 
def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)






def Wk(mu, clusters):
    K = len(mu)
    return sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
               for i in range(K) for c in clusters[i]])


def bounding_box(X):
    xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
    ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
    return (xmin,xmax), (ymin,ymax)

def gap_statistic(X):
    (xmin,xmax), (ymin,ymax) = bounding_box(X)
    # Dispersion for real distribution
    ks = range(1,11)
    Wks = np.zeros(len(ks))
    Wkbs = np.zeros(len(ks))
    sk = np.zeros(len(ks))
    for indk, k in enumerate(ks):
        mu, clusters = find_centers(X,k)
        Wks[indk] = np.log(Wk(mu, clusters))
        # Create B reference datasets
        B = 10
        BWkbs = np.zeros(B)
        for i in range(B):
            Xb = []
            for n in range(len(X)):
                Xb.append([random.uniform(xmin,xmax),
                          random.uniform(ymin,ymax)])
            Xb = np.array(Xb)
            mu, clusters = find_centers(Xb,k)
            BWkbs[i] = np.log(Wk(mu, clusters))
        print "Task percentage : ",(k/40.0)*100,"%\n"
        Wkbs[indk] = sum(BWkbs)/B
        sk[indk] = np.sqrt(sum((BWkbs-Wkbs[indk])**2)/B)
    sk = sk*np.sqrt(1+1/B)
    return(ks, Wks, Wkbs, sk)




def init_board_gauss(N, k):
    n = float(N)/k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    return X



'''

X = init_board_gauss(100,1)
xl=[x[0] for x in X]
yl=[x[1] for x in X]
xl=np.array(xl)
yl=np.array(yl)


# figure 1
plt.figure()
plt.xlabel("x"),plt.ylabel("y")
plt.plot(xl,yl,'.')

ks, logWks, logWkbs, sk = gap_statistic(X)
gaps=logWkbs-logWks

#figure 2
plt.figure()
plt.xlabel("Number of clusters K"), plt.ylabel("Gap")
plt.plot(ks, gaps)

#figure 3
plt.figure()
plt.xlabel("Number of clusters K")
plt.plot(ks, logWks, 'r')
plt.plot(ks, logWkbs, 'b')





ll = len(gaps)
result = np.zeros(ll)
for i in range(0, ll - 1):
    result[i] = gaps[i] - gaps[i+1] - sk[i+1]
print result 
#figure 4
plt.figure()
plt.xlabel("Number of clustres K"), plt.ylabel("Gap(k)-Gap(k+1)-sk(k+1)")
plt.bar(ks, result)

plt.show()







'''
fileName = raw_input("Enter the file's name")
print "Reading dataset\n"

#Points' coordinate
location = []

label = []
#Read coordinate from the data set
for line in open(fileName, "r"):
    items = line.strip("\n").split(",")
    label.append(int(items.pop()))
    tmp = []
    for item in items:
        tmp.append(float(item))
    location.append(tmp)
location = np.array(location)
label = np.array(label)
length = len(location)
print "Reading dataset finished\n"

print "Calculating Gaps\n"
ks, logwks, logwkbs, sk = gap_statistic(location)
gaps=logwkbs-logwks
print "Cal finished\n"
print "Drawing\n"



R = range(256)
random.shuffle(R)
R = np.array(R)/255.0
G = range(256)
random.shuffle(G)
G = np.array(G)/255.0
B = range(256)
random.shuffle(B)
B = np.array(B)/255.0
colors = []
for i in range(256):
    colors.append((R[i], G[i], B[i]))


plt.figure()
for i in range (length):
	index = label[i]
	plt.plot(location[i][0],location[i][1],color = colors[index], marker = '.')
plt.show()


plt.figure()
gaps = logwkbs - logwks
plt.plot(ks, gaps)
plt.show()


ll = len(gaps)
result = np.zeros(ll)
for i in range(0, ll - 1):
    result[i] = gaps[i] - gaps[i+1] - sk[i+1]
print result 
#figure 4
plt.figure()
plt.xlabel("Number of clustres K"), plt.ylabel("Gap(k)-Gap(k+1)-sk(k+1)")
plt.bar(ks, result)

plt.figure()
plt.plot(ks, gaps)
plt.show()















