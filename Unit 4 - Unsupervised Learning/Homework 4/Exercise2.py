import numpy as np
import utils
seq = 'A B A B B C A B A A B C A C'
seq = seq.replace(" ",'')

#region Unigram Model
## item 1
print('------------------ Unigram Model ---------------------\n')

unigram = sorted(set(seq))
dict_th = utils.ngram_mle(seq,unigram)

## item 2
test_seq = ['ABC', 'BBB', 'ABB', 'AAC']
mle_dict = {}
for tq in test_seq:
    mle_dict[tq] = utils.mle(dict_th, tq)

print('\nMLE Estimate(D): ',mle_dict)
print('------------------------------------------------------\n')
#endregion

#region Bigram Model
print('-------------------- BIGRAM ------------------------\n')
bigrams = sorted([l+k for l in set(seq) for k in set(seq)])
dict_th2 = utils.ngram_mle(seq,bigrams)

test_seq = "A A B C B A B"
test_seq = test_seq.replace(' ','')

mle = utils.mle(dict_th2,test_seq)
print('\nMLE Estimate(D): ',mle)
print('------------------------------------------------------\n')

#endregion

