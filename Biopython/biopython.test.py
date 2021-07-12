#!/usr/bin/python

### 1. how it works
# get some data in stockholm format
# read file with `Bio.AlignIO`
from Bio import AlignIO

alignment = AlignIO.read(open("PF1234_seed.txt"), "stockholm")
print(alignment)        # to show the file content

for align in alignment:
    print(align.seq)

### 2. multi alignment
from Bio import AlignIO
alignments = AlignIO.parse(open("PF1234_seed.txt"), "stockholm")
print(alignments)

for alignment in alignments:
    print(alignment)

### 3. paired seq align
from Bio import pairwise2
from Bio.Seq import Seq     # to create 2 seqs
seq1 = Seq("ATCCTGA")
seq2 = Seq("ATTCAG")

alignments = pairwise2.align.globalxx(seq1,seq2)
test_alignments = pairwise2.align.localds(seq1, seq2, blosum62, -10, -1)
for alignment in alignments:
    print(alignment)
``` output info:
('ACCGGT', 'A-C-GT', 4.0, 0, 6)
('ACCGGT', 'AC--GT', 4.0, 0, 6)
('ACCGGT', 'A-CG-T', 4.0, 0, 6)
('ACCGGT', 'AC-G-T', 4.0, 0, 6)
```

from Bio.pairwise2 import format_alignment
alignments = pairwise2.align.globalxx(seq1, seq2)
for alignment in alignments:
    print(format_alignment(*alignment))
``` output info:
ACCGGT
| | ||
A-C-GT
   Score=4

ACCGGT
|| ||
AC--GT
   Score=4
```
