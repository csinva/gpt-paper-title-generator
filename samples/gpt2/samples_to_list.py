import os

all_samples = []
d = 'run1'
for fname in sorted(os.listdir(d), reverse=True):
    if not fname.startswith('samples'):
        continue
    s = open(d + '/' + fname, 'r').read()
    s = s.replace('<|endoftext|>', '\n')
    s = s.replace('startoftext', '').replace('endoftext', '').replace('<', '').replace('>', '').replace('|', '') #.title()
    l = s.split('\n')[1:]
    all_samples += l
all_samples = [a for a in all_samples
               if not '===' in a
               and not a == ''
               and not 'http' in a # no links
               and not len(a.split()) < 2 # only one word
              ]

with open(os.path.join(d, 'all.txt'), 'w') as f:
    f.write('\n\n'.join(all_samples))
