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


def create_data_and_designmatrix(img,
                                 N=500,
                                 width=10,
                                 p=1.0,
                                 deltaa=None,
                                 ixrand=None,
                                 iyrand=None,
                                 Mx=None,
                                 My=None):
    if Mx is None or My is None:
        Mx = img.shape[0]
        My = img.shape[1]
    if ixrand is None or iyrand is None:
        ixrand = np.array(list(map(int,
                                   np.random.rand(N) * Mx *
                                   p)))  # x-position of beams
        iyrand = np.array(list(map(int,
                                   np.random.rand(N) *
                                   My)))  # y-position of beams
    if deltaa is None:
        deltaa = np.array(list(map(int,
                                   np.random.rand(Mx * My) * width))).reshape(
                                       Mx, My)  #width of beams

    d = []
    g = []
    imgext = np.zeros((Mx, My))
    imgext[:, :] = None

    for i in range(0, N):
        ix = ixrand[i]
        iy = iyrand[i]
        delta = deltaa[ix, iy]
        ix1 = max(ix - delta, 0)
        ix2 = min(Mx, ix + delta)
        iy1 = max(iy - delta, 0)
        iy2 = min(My, iy + delta)
        val = np.sum(img[ix1:ix2, iy1:iy2])
        d.append(val)
        gzero = np.zeros((Mx, My), dtype="float")
        gzero[ix1:ix2, iy1:iy2] = 1.0
        g.append(gzero.flatten())
        imgext[ix, iy] = np.mean(img[ix1:ix2, iy1:iy2])

    d = np.array(d)
    g = np.array(g)

    return d, g, imgext


def plot_lcurve(residualseq, modelnormseq, curveseq, imax):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    pylab.xscale('log')
    pylab.yscale('log')
    pylab.ylabel("Norm of Model")
    pylab.xlabel("Norm of Prediction Error")
    ax.plot(residualseq, modelnormseq, marker=".", c="green")
    ax.plot([residualseq[imax]], [modelnormseq[imax]], marker="o", c="red")

    ax2 = fig.add_subplot(122)
    pylab.xscale('log')
    pylab.ylabel("Curvature")
    pylab.xlabel("Norm of Prediction Error")
    ax2.plot(residualseq, curveseq, marker=".", c="green")
    ax2.plot([residualseq[imax]], [curveseq[imax]], marker="o", c="red")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Inverse Problem test program by Hajime Kawahara')
    parser.add_argument('-f', nargs=1, required=True, help='png file')
    parser.add_argument('-n',
                        nargs=1,
                        default=[1000],
                        help='number of light beams',
                        type=int)
    parser.add_argument('-l',
                        nargs=1,
                        default=[0.001],
                        help='lambda',
                        type=float)
    parser.add_argument('-L',
                        nargs=2,
                        help='L curve criterion ON. Input search area',
                        type=float)
    parser.add_argument('-nl',
                        nargs=1,
                        default=[40],
                        help='number of lambda grid for L curve',
                        type=int)
    parser.add_argument('-w',
                        nargs=1,
                        default=[20.0],
                        help='mean beam diameter',
                        type=float)
    parser.add_argument('-p',
                        nargs=1,
                        default=[1.0],
                        help='beams probe on upper 100 p percent area ',
                        type=float)
    parser.add_argument(
        '-lim',
        nargs=1,
        default=[1.e-12],
        help=
        'NGIM or TSVD limit (singular values below this are regarded as zero)',
        type=float)
    parser.add_argument('-s',
                        nargs=1,
                        help='STD of gaussian noise',
                        type=float)
    parser.add_argument('-save',
                        nargs=1,
                        help='save G and d of search light',
                        type=str)
    parser.add_argument('-load',
                        nargs=1,
                        help='load G and d of search light',
                        type=str)
    parser.add_argument('-solver',
                        nargs=1,
                        default=["fullsvd"],
                        help='SVD solver. fullsvd or iterative',
                        type=str)
    args = parser.parse_args()

    solver = args.solver[0]
    img = rpng.get_bwimg(args.f[0])
    lamb = args.l[0]
    width = args.w[0]
    N = args.n[0]
    p = args.p[0]
    M = Mx * My
    print("# of Data = ", N, "# of Model = ", M)

    if args.load:
        #load d,g,imgext
        print("Load G and d from ", args.load[0])
        data = np.load(args.load[0] + ".npz")
        d = data["arr_0"]
        g = data["arr_1"]
        imgext = data["arr_2"]
    else:
        #create eclipse curves
        d, g, imgext = create_data_and_designmatrix(img, N, width, p)

        if args.s:
            sigma = args.s[0]
            print("noise injection to data: sigma=", sigma)
            d = d + np.random.normal(0.0, sigma, N)

    if args.save:
        print("Save G and d to ", args.save[0])
        np.savez(args.save[0], d, g, imgext)

    print("compute svd")
    if solver == "fullsvd":
        start = time.time()
        U, S, VT = np.linalg.svd(g)
        elapsed_time = time.time() - start
        print("solved by np.linalg.svd: time=", elapsed_time)
        p = None
    elif solver == "iterative":
        nk = min(M, N) - 1
        print(nk, M, N)
        start = time.time()
        U, S, VT = scipy.sparse.linalg.svds(g, k=nk)
        elapsed_time = time.time() - start
        #convert values in decending order
        S = S[::-1]
        U = U[:, ::-1]
        VT = VT[::-1, :]
        #remove nan
        mask = (S == S)
        S = S[mask]
        U = U[:, mask]
        VT = VT[mask, :]
        print("solved by scipy.sparse.linalg.svds: time=", elapsed_time)
        p = len(S)
    else:
        sys.exit("invalid solver option. specify fullsvd or iterative. EXIT.")

    mprior = np.zeros(M)

    if args.L:
        method = "Choose adequate lambda by L-curve criterion"
        print(method)
        modelnormseq = []
        residualseq = []
        curveseq = []
        nlcurve = args.nl[0]
        lamseq = np.logspace(np.log10(args.L[0]),
                             np.log10(args.L[1]),
                             num=nlcurve)
        print("lamb", "curvature")
        for lamb in lamseq:
            mest, dpre, residual, modelnorm, curv_lcurve = tik.tikhonov_regularization(
                g, d, mprior, U, VT, S, lamb, p=p)
            modelnormseq.append(modelnorm)
            residualseq.append(residual)
            curveseq.append(curv_lcurve)
            print(lamb, curv_lcurve)

        residualseq = np.array(residualseq)
        modelnormseq = np.array(modelnormseq)
        imax = np.argmax(curveseq)
        lamb = lamseq[imax]
        print("Best lambda=", lamb)
        plot_lcurve(residualseq, modelnormseq, curveseq, imax)

    if lamb == 0:
        method = "NGIM/TSVD"
        lim = args.lim[0]
        print(method)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(0, len(S), 1), S, ".")
        pylab.yscale("log")
        pylab.axhline(lim, color="red")
        pylab.xlabel("i")
        pylab.ylabel("Singular Value")
        plt.show()

        mest, dpre, residual, modelnorm = tik.NGIM_regularization(
            g, d, mprior, U, VT, S, lim)
        cap = "# of beams = " + str(N) + ", Mean beam diameter = " + str(
            width) + " pixels," + " singular value cutoff = " + str(lim)

    else:
        method = "Tikhonov regularization"
        print(method)
        mest, dpre, residual, modelnorm, curv_lcurve = tik.tikhonov_regularization(
            g, d, mprior, U, VT, S, lamb, p=p)
        cap = "# of beams = " + str(N) + ", Mean beam diameter = " + str(
            width) + " pixels, $\lambda$ = " + str(lamb)

    imgest = mest.reshape(Mx, My)

    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.imshow(img, cmap="gray")
    pylab.title("input")
    ax = fig.add_subplot(132)
    ax.imshow(imgext, cmap="gray")
    pylab.title("data (averaged)")
    ax.annotate(method,
                xy=(0.5, 1.15),
                xycoords='axes fraction',
                horizontalalignment="center",
                fontsize=16)
    ax.annotate(cap,
                xy=(0.5, -0.2),
                xycoords='axes fraction',
                horizontalalignment="center",
                fontsize=12,
                color="gray")
    ax = fig.add_subplot(133)
    ax.imshow(imgest, cmap="gray")
    pylab.title("estimate")

    plt.show()
