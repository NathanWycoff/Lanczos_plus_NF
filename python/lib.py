
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/toy_kl_div.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 05.05.2019

## Practice doing VB with MC on an intractable ELBO.
# x ~ N(0,I); f(x) = Ax; find A such that f(x) best matches N(0, SIGMA) for some SIGMA.

#TODO: Do true SGD on it (rn our batch size is so large it may as well be GD).

def smin_kl(rho_samp, log_post, transformer):
    """
    Stochastically minimize KL divergence between the variational dist and a posterior.

    :param rho_samp: The base measure associated with the variational distribution. rho_samp should accept 2 scalar integer arguments specifying how many draws to return and the dimension of the space, that is rho_samp(B,P) should return B many samples in P dimensional space.
    :param log_post: Given a P-vector, this should return the 
    """
