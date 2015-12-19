# pinvprob
Python codes for the linear inverse problem including the generalized inverse matrix, truncated SVD, Tikhonov regularization, L-curve criterion

**Version 0.1, Hajime Kawahara**

Originally, I developed these codes for two papers (Fortran 90), 
Kawahara & Fujii (2011)  http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1106.0136
Fujii & Kawahara (2012)  http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1204.3504
. I imported them to Python for the internal seminar of our university.

For the L-curve criterion, see the brilliant book, Hansen, P. C. 2010, Discrete Inverse Problems: Insight and Algorithms (the Society for Industrial and Applied Mathema
tics).

I made "random_light.py" for a demonstration of solving the inverse problem. This demonstration retrieves a small png image from a collection of summation of random rectangle parts of the image. If you use Japanese, see invprov.pdf, otherwise see Figure 1 in invprov.pdf, you will understand this problem. 

Note that these codes are inefficient to deal with a large image because the codes directly uses the singular value decomposition. The sample image che.png was taken from Wikipedia.

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

* Use the L-curve Criterion to search an appropriate regularization parameter. 

~~~~
 ./random_light.py -f che.png -n 2500 -L 0.01 100.0 -p 0.7 -s 1.0
~~~~
