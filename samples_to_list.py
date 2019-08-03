import os

all = []
d = 'samples_bio'
for fname in os.listdir(d):
    s = open(d + '/' + fname, 'r').read()
    s = s.replace('startoftext', '').replace('endoftext', '').replace('<', '').replace('>', '').replace('|', '').title()
    l = s.split('\n')[1:]
    all += l
all = [a for a in all if not '===' in a and not a == '']

print(all)
