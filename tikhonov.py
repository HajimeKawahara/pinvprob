def tikhonov_regularization(Dm,Ao,Smprior,U,VT,S,lambdatik,p=None):
    import numpy as np
    #should be np.array
    sumphi=0.0
    dxidlam=0.0

    Ndata=len(Ao)
    Mmodel=len(Smprior)
    Sm=np.zeros(Mmodel) 

    dv=[]
    for idata in range(0,Ndata):
#       dv.append(Ao(idata)-DOT_PRODUCT(array(Dm(idata,1:Mmodel)),Smprior(1:Mmodel)))
        dv.append(Ao[idata]-np.inner(Dm[idata,:],Smprior))

###
    if p is None:
        p=min(Ndata,Mmodel)
    
    for i in range(0,p):
           phii=(S[i]**2/(S[i]**2+lambdatik**2))
           phij=(S[i]/(S[i]**2+lambdatik**2))
           dxidlam=dxidlam+(1.0-phii)*(phij**2)*(np.inner(U[:,i],dv)**2)
           Sm=Sm+phij*np.inner(U[:,i],dv)*VT[i,:]
           
    dxidlam=-4.0*dxidlam/lambdatik

    Sm=Sm+Smprior    
    Aoest=np.dot(Dm,Sm)
    residual=np.linalg.norm(Ao-Aoest)
    modelnorm=np.linalg.norm(Sm-Smprior)

    lam2=lambdatik**2
    rho=residual*residual
    xi=modelnorm*modelnorm
    curv_lcurve=-2.0*xi*rho/dxidlam*(lam2*dxidlam*rho+2.0*lambdatik*xi*rho+lam2*lam2*xi*dxidlam)/((lam2*lam2*xi*xi+rho*rho)**1.5)

    return Sm,Aoest,residual,modelnorm,curv_lcurve

def NGIM_regularization(Dm,Ao,Smprior,U,VT,S,lim):
    import numpy as np
    #should be np.array

    p = np.sum(S>lim)
    print "NGIM/TSVD: we regard values below ",lim," as zero."
    print "p = ", p
    sumphi=0.0

    Ndata=len(Ao)
    Mmodel=len(Smprior)
    Sm=np.zeros(Mmodel) 

    dv=[]
    for idata in range(0,Ndata):
        dv.append(Ao[idata]-np.inner(Dm[idata,:],Smprior))

    for i in range(0,p):
        phij=(1.0/S[i])
        Sm=Sm+phij*np.inner(U[:,i],dv)*VT[i,:]

    Sm=Sm+Smprior    
    Aoest=np.dot(Dm,Sm)
    residual=np.linalg.norm(Ao-Aoest)
    modelnorm=np.linalg.norm(Sm-Smprior)

    return Sm,Aoest,residual,modelnorm
