#!/usr/bin/python
import pylab
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import readpng as rpng
import tikhonov as tik
import scipy.sparse.linalg
import time
import random_light as rl
import sklearn.linear_model as lm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inverse Problem test program by Hajime Kawahara using sklearn.Ridge')
    parser.add_argument('-f', nargs=1, required=True, help='png file')
    parser.add_argument('-n', nargs=1, default=[1000], help='number of light beams', type=int)
    parser.add_argument('-l', nargs='+', default=[0.001], help='lambda. if you specify multiple lambdas, then the cross validation is performed. ', type=float)
    parser.add_argument('-w', nargs=1, default=[20.0], help='mean beam diameter', type=float)
    parser.add_argument('-p', nargs=1, default=[1.0], help='beams probe on upper 100 p percent area ', type=float)
    parser.add_argument('-s', nargs=1, help='STD of gaussian noise', type=float)
    parser.add_argument('-save', nargs=1, help='save G and d of search light', type=str)
    parser.add_argument('-load', nargs=1, help='load G and d of search light', type=str)
    parser.add_argument('-solver', nargs=1, default=["sklearn"], help='SVD solver. sklearn.Ridge, fullsvd, lasso', type=str)
    args = parser.parse_args()    

    solver=args.solver[0]
    img=rpng.get_bwimg(args.f[0])
    if len(args.l)==1:
        lamb=args.l[0]
    else:
        lamb=0.0
        lamblist=args.l
    width=args.w[0]
    N=args.n[0]
    p=args.p[0]
    Mx=img.shape[0]
    My=img.shape[1]
    M=Mx*My
    print("# of Data = ", N, "# of Model = ", M)

    if args.load:
    #load d,g,imgext
        print("Load G and d from ",args.load[0])
        data=np.load(args.load[0]+".npz")
        d=data["arr_0"]
        g=data["arr_1"]
        imgext=data["arr_2"]
    else:
    #create eclipse curves
        deltaa=np.array(list(map(int,np.random.rand(Mx*My)*width))).reshape(Mx,My) #width of beams       
        ixrand=np.array(list(map(int,np.random.rand(N)*Mx*p))) # x-position of beams       
        iyrand=np.array(list(map(int,np.random.rand(N)*My)))   # y-position of beams       
        d,g,imgext=rl.create_data_and_designmatrix(N,Mx,My,ixrand,iyrand,deltaa,img)
        
        if args.s:
            sigma=args.s[0]
            print("noise injection to data: sigma=",sigma)
            d=d+np.random.normal(0.0,sigma,N) 

    if args.save:
        print("Save G and d to ",args.save[0])
        np.savez(args.save[0],d,g,imgext)

        
    if solver == "fullsvd":
        start = time.time()
        U,S,VT=np.linalg.svd(g)
        p=None
        mprior=np.zeros(M)
        mest,dpre,residual,modelnorm,curv_lcurve=tik.tikhonov_regularization(g,d,mprior,U,VT,S,lamb,p=p)
        imgest=mest.reshape(Mx,My)
        elapsed_time = time.time() - start
        print("solved by np.linalg.svd: time=",elapsed_time)
        method="Tikhonov Regularization"
    elif solver == "sklearn":
        start = time.time()
        if lamb > 0.0:
            clf = lm.Ridge(alpha = lamb)
            clf.fit(g,d)  
        else:
            print("Cross Validation between ",lamblist)
            clf = lm.RidgeCV(alphas = lamblist)
            print(lamblist)
            clf.fit(g,d)  
            lamb=clf.alpha_ 
            print("Result: lambda=",lamb)
        mest=clf.coef_
        dpre=np.dot(g,mest)+clf.intercept_ 
        imgest=mest.reshape(Mx,My)
        elapsed_time = time.time() - start
        print("solved by scikit_learn.Ridge: time=",elapsed_time)
        method="Tikhonov Regularization"
    elif solver == "lasso":
        start = time.time()
        if lamb > 0.0:
            clf = lm.Lasso(alpha = lamb)
            clf.fit(g,d)  
        else:
            print("Cross Validation between ",lamblist)
            clf = lm.LassoCV(alphas = lamblist)
            print(lamblist)
            clf.fit(g,d)  
            lamb=clf.alpha_ 
            print("Result: lambda=",lamb)
        mest=clf.coef_
        dpre=np.dot(g,mest)+clf.intercept_ 
        imgest=mest.reshape(Mx,My)
        elapsed_time = time.time() - start
        print("solved by scikit_learn.Lasso: time=",elapsed_time)
        method="LASSO"

    else:
        sys.exit("invalid solver option. specify sklearn or fullsvd. EXIT.")
    cap="# of beams = "+str(N)+", Mean beam diameter = "+str(width)+" pixels, $\lambda$ = "+str(lamb)


    fig =plt.figure()
    ax=fig.add_subplot(131)
    ax.imshow(img,cmap="gray")
    pylab.title("input")
    ax=fig.add_subplot(132)
    ax.imshow(imgext,cmap="gray")
    pylab.title("data (averaged)")
    ax.annotate(method, xy=(0.5, 1.15), xycoords='axes fraction',horizontalalignment="center", fontsize=16)
    ax.annotate(cap, xy=(0.5, -0.2), xycoords='axes fraction',horizontalalignment="center", fontsize=12,color="gray")
    ax=fig.add_subplot(133)
    ax.imshow(imgest,cmap="gray")
    pylab.title("estimate")

    plt.show()

