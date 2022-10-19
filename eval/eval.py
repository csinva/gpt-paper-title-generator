# BLEU code

import nltk
import nltk.translate.bleu_score as bleu

import math
import numpy
import os

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def title_bleu(tA, tB):
    return bleu.sentence_bleu([tA.lower().split()], tB.lower().split())

if __name__ == '__main__':
    title_bleu(
        'Neuromorphic transformer diffusion kernel models for the creation of buzzwords',
        'Existential neuromorphic transformer graph models for the creation of buzzwords'
    )

    # 0.5814307369682193
