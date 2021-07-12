from Bio import SeqIO

records = list(SeqIO.parse("ap006852.fasta", "fasta"))
dna = records[0]

print(dna.name)
print(dna.description)
print(dna.seq[:100])

# or this way
from Bio import SeqIO
for record in SeqIO.parse("ls_orchid.fasta", "fasta"):
    print record.seq, len(record.seq)
