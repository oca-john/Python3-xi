import Bio
from Bio.Seq import Seq
dna = Seq("ACGTTGCAC")
print(dna)

# or this way
from Bio.Alphabet import IUPAC
dna = Seq("AGTACACTGGT", IUPAC.unambiguous_dna)
