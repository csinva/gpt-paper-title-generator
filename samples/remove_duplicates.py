import numpy as np
import os

fnames = [f for f in os.listdir('.') if f.startswith('samples_')]
orig = open('titles.csv', 'r').read().split('\n')
orig_nospace = ["".join(s.split()) for s in orig]
print(orig_nospace)
samps_tot = []
for fname in fnames:
    samps = open(fname, 'r').read()
    samps = samps.split('\n')
    samps = [s.replace('<|endoftext|>', '')
             .replace('<|startoftext|>', '')
             .replace('startoftext', '')             
             .replace('endoftext', '')               
             .replace('<', '')                          
             .replace('|', '')    
             .replace('>', '')    
             .replace('<', '')    
             for s in samps
             ]
    samps = [s for s in samps
             if len(s) > 1
             and not "".join(s.split()) in orig_nospace]
    samps_tot += samps

with open('samples/samples_wolf_tweets/all.txt', 'a') as f:
    f.write('\n\n'.join(samps_tot))