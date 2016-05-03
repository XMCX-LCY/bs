import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random

def plot1(rho, delta):
    f, axarr = plt.subplots(1,3)
    axarr[0].set_title('DECISION GRAPH')
    axarr[0].scatter(rho, delta, alpha=0.6,c='white')
    axarr[0].set_xlabel(r'$\rho$')
    axarr[0].set_ylabel(r'$\delta$')
    axarr[1].set_title('DECISION GRAPH 2')
    axarr[1].scatter(np.arange(len(rho))+1, -np.sort(-rho*delta), alpha=0.6,c='white')
    axarr[1].set_xlabel('Sorted Sample')
    axarr[1].set_ylabel(r'$\rho*\delta$')
    return(f,axarr)

def plot2(axarr,rho, delta,cmap,cl,icl,XY,NCLUST):
    
    #axarr.set_title('2D multidimensional scaling')
    axarr.plot( XY[:,0],  XY[:,1],alpha=0.8,c=cmap[list(cl)])
    axarr.xlabel('X')
    axarr.ylabel('Y')


def DCplot(dist, XY, ND, rho, delta, ordrho, dc, nneigh, rhomin, deltamin, num):
    
    lamb = rho * delta 
    ordlamb = (-lamb).argsort()
    cl = np.zeros(ND)-1
    icl = np.zeros(1000)
    NCLUST = 0

    for i in range(num):
        index = ordlamb[i]
        cl[index] = NCLUST
        icl[NCLUST] = index
        NCLUST = NCLUST + 1

    print('NUMBER OF CLUSTERS: %i'%(NCLUST))
    print('Performing assignation')
    #assignation
    for i in range(ND):
        if cl[ordrho[i]]==-1:
            cl[ordrho[i]] = cl[nneigh[ordrho[i]]]

    #halo
    # cluster id start from 1, not 0
    ## deep copy, not just reference
    halo = np.zeros(ND)
    halo[:] = cl

    if NCLUST>1:
        bord_rho = np.zeros(NCLUST)
        for i in range(ND-1):
            for j in range((i+1),ND):
                if cl[i]!=cl[j] and dist[i,j]<=dc:
                    rho_aver = (rho[i]+rho[j])/2
                    if rho_aver>bord_rho[cl[i]]:
                        bord_rho[cl[i]] = rho_aver
                    if rho_aver>bord_rho[cl[j]]:
                        bord_rho[cl[j]] = rho_aver
        for i in range(ND):
            if rho[i]<bord_rho[cl[i]]:
                halo[i] = -1

    for i in range(NCLUST):
        nc = 0
        nh = 0
        for j in range(ND):
            if cl[j]==i:
                nc = nc+1
            if halo[j]==i:
                nh = nh+1
        print('CLUSTER: %i CENTER: %i ELEMENTS: %i CORE: %i HALO: %i'%( i+1,icl[i]+1,nc,nh,nc-nh))
            # print , start from 1

            ## save CLUSTER_ASSIGNATION
    print('Generated file:CLUSTER_ASSIGNATION')
    print('column 1:element id')
    print('column 2:cluster assignation without halo control')
    print('column 3:cluster assignation with halo control')
    clusters = np.array([np.arange(ND)+1,cl+1,halo+1]).T
    np.savetxt('CLUSTER_ASSIGNATION_%.2f_%.2f_.txt'%(rhomin,deltamin),clusters,fmt='%d\t%d\t%d')
    print('Result are saved in file CLUSTER_ASSIGNATION_%.2f_%.2f_.txt'%(rhomin,deltamin))
    print('\n\nDrag the mouse pointer at a cutoff position in figure DECISION GRAPH and press   OR   Press key n to quit')
    ################# plot the data points with cluster labels
    cmap = cm.rainbow(np.linspace(0, 1, NCLUST))
    #plot2(ax,rho, delta,cmap,cl,icl,XY,NCLUST)

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
    for i in range(ND):
        index = int(cl[i])
        plt.plot(XY[i][0],XY[i][1], color = colors[index], marker = '.')
    plt.show()
  
    

    
 
