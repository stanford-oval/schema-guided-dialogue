from sacrebleu import corpus_bleu
import csv

def computeBLEU(outputs, targets):
    targets = [[t[i] for t in targets] for i in range(len(targets[0]))]
    return corpus_bleu(outputs, targets, smooth_method="exp",
                                     smooth_value=0.0,
                                     force=False,
                                     lowercase=False,
                                     tokenize="intl",
                                     use_effective_order=False).score

pathname = 'test.tsv'
tsv_file = open(pathname)
read_tsv = csv.reader(tsv_file, delimiter="\t")


cpy = [row for row in read_tsv]
outputs = [row[0] for row in cpy]
targets = [[row[1]] for row in cpy]

assert(len(outputs) == len(targets))
bleu = computeBLEU(outputs, targets)
print(bleu)