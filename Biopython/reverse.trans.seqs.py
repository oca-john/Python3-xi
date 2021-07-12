dna.reverse_complement()  # 反向互补
rna = dna.transcribe()    # 转录
rna.translate()           # 翻译

# or this way
from Bio.Seq import reverse_complement, transcribe, translate
reverse_complement("GCTGTTATGGGTCGTTGGAAGGGTGGTCGTGCT")
