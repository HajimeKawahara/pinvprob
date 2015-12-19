pinvprob
=========


[![Licence](http://img.shields.io/badge/license-GPLv2-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-2.0.html)
[![arXiv](http://img.shields.io/badge/arXiv-1106.0136-blue.svg?style=flat)](http://arxiv.org/abs/1106.0136)
[![arXiv](http://img.shields.io/badge/arXiv-1204.3504-blue.svg?style=flat)](http://arxiv.org/abs/1204.3504)

Python codes for the linear inverse problem including the generalized inverse matrix, truncated SVD, Tikhonov regularization, L-curve criterion

**Version 0.1, Hajime Kawahara**

Originally, I developed these codes for two papers (Fortran 90), 
Kawahara & Fujii (2011)  http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1106.0136
Fujii & Kawahara (2012)  http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1204.3504
. I imported them to Python for the internal seminar of our university.

For the L-curve criterion, see the brilliant book, Hansen, P. C. 2010, Discrete Inverse Problems: Insight and Algorithms (the Society for Industrial and Applied Mathema
tics).

I made "random_light.py" for a demonstration of solving the inverse problem. This demonstration retrieves a small png image from a collection of summation of random rectangle parts of the image. If you use Japanese, see invprov.pdf, otherwise see Figure 1 in invprov.pdf, you will understand this problem. 

Note that these codes are inefficient when the image size is large because the codes directly use the singular value decomposition. The sample image che.png was taken from Wikipedia and was compressed to a small image.

Requirements
------------------

* python 2.7
* scipy
* pylab


Tutorial
-------------------------

* Solve the problem by the Natural Generalized Inverse Matrix (NGIM) with no noise.

~~~~
 ./random_light.py -f che.png -n 1000 -l 0.0 -p 0.7 -w 20.0
~~~~

* Solve the problem by the Natural Generalized Inverse Matrix (NGIM) with an additional noise. The retrieved map is very unstable.

~~~~ 
 ./random_light.py -f che.png -n 2500 -l 0.0 -p 0.7 -s 1.0
~~~~

* Solve the problem by the Tiknov regularization with an additional noise. 

~~~~
 ./random_light.py -f che.png -n 2500 -l 3.0 -p 0.7 -s 1.0
~~~~

* Solve the problem by the Truncated Singular Value Decomposition (TSVD) with an additional noise. 

~~~~
 ./random_light.py -f che.png -n 2500 -l 0.0 -p 0.7 -s 1.0 -lim 1.0
~~~~

* Use the L-curve criterion (Hansen 2010) to look for an appropriate regularization parameter. 

~~~~
 ./random_light.py -f che.png -n 2500 -L 0.01 100.0 -p 0.7 -s 1.0
~~~~

License
------------

GPL. See "License" in detail. 