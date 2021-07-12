from Bio import SeqIO

records = list(SeqIO.parse("ap006852.fasta", "fasta"))
dna = records[0]

print(dna.name)
print(dna.description)
print(dna.seq[:100])
