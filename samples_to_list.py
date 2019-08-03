import os

all = []
for fname in os.listdir('samples'):
    s = open('samples/' + fname, 'r').read()
    s = s.replace('startoftext', '').replace('endoftext', '').replace('<|', '').replace('|>', '').title()
    l = s.split('\n')
    l.remove('')
    all += l
all = [a for a in all if not '===' in a]

print(all)
