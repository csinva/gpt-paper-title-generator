import os

all_samples = []
d = 'samples_wolf_tweets'
for fname in os.listdir(d):
    if not fname.startswith('samples'):
        continue
    s = open(d + '/' + fname, 'r').read()
    s = s.replace('<|endoftext|>', '\n')
    s = s.replace('startoftext', '').replace('endoftext', '').replace('<', '').replace('>', '').replace('|', '') #.title()
    l = s.split('\n')[1:]
    all_samples.append(l)
all_samples = [a for a in all_samples
               if not '===' in a and not a == '']

print(all_samples)
