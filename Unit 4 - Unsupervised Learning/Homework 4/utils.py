import numpy as np

def ngram_mle(seq, ngrams):
    dict_count = {}
    dict_th = {}
    n = len(ngrams[0])

    for ng in ngrams:
        dict_count[ng] = seq.count(ng)
        if n == 1: C = len(seq)
        else: C = seq.count(ng[0:n-1])
        dict_th[ng] = dict_count[ng] / C
        print('Estimate ' + ng + ': ', dict_th[ng])
    return dict_th

def mle(params, test_seq):

    mle = 1
    n = len(list(params.keys())[0])
    for j in range(len(test_seq)-n+1):
        ngram = test_seq[j:j+n]
        mle *= params[ngram]

    return mle

def multi_normal(mu,var,x):

    d = 1 if not hasattr(mu, '__iter__') else x.shape[1]
    const = 1/((2*np.pi*var)**d/2)
    exp_x = ((x-mu)**2)#.reshape(-1,1)
    exp = np.exp(-1/2*exp_x/var)
    return (const*exp)
