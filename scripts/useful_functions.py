import os
import pyperclip
import random
import math
import pandas as pd
from Bio import SeqIO
from typing import List, Dict, Tuple

def add_counts(df: pd, colname: str, outname: str) -> pd:
    counts = df.groupby(colname).size().reset_index(name=outname)
    # Merge the counts into the original dataframe
    df = df.merge(counts, on=colname)
    return df

def remove_char(s: str, c: str) -> str:
    if not s:  # base case: if the string is empty, return an empty string
        return ""
    if s[0] == c:  # if the first character of the string is the character to remove, skip it and recurse on the rest of the string
        return remove_char(s[1:], c)
    else:  # if the first character is not the character to remove, keep it and recurse on the rest of the string
        return s[0] + remove_char(s[1:], c)


def translate_dna(sequence):
    from Bio.Seq import Seq
    dna_seq = Seq(sequence)
    prot_seq = dna_seq.translate()
    protein = str(prot_seq)
    return protein


# prime number calculator
def prime_generator(n: int) -> List[int]:
    l = []
    for i in range(0, n + 1):
        count = 0
        for j in range(1, i + 1):
            if i % j == 0:
                count += 1
        if count <= 2:
            if i not in l:
                l.append(i)
    return l

def is_it_prime_efficient(n: int) -> int:
    count = 0
    for i in range(1, n + 1):
        if n % i == 0:
            count += 1
        if count > 2:
            return 0
    return 1

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size

def excel_to_list() -> list:
    s = pyperclip.paste()
    return s.split('\r\n')

def random_oligo(length: int) -> str:
    bases = ['A', 'C', 'G', 'T']
    return ''.join(random.choices(bases, k=length))

def random_oligo_plate(plate_size: int, oligo_length: int) -> List[str]:
    '''generates a plate of random, unique oligos - application is for multiplexing sanger samples for NGS'''
    l = []
    for i in range(plate_size):
        p = random_oligo(oligo_length)
        while p in l:
            p = random_oligo(oligo_length)
        l.append(random_oligo(oligo_length))
    return l


def oligo_calc_salt_adjust(dna_seq: str) -> tuple[float, float, int]:
    '''Tm= (wA+xT)*2 + (yG+zC)*4 - 16.6*log10(0.050) + 16.6*log10([Na+])
    where w,x,y,z are the number of the bases A,T,G,C in the sequence, respectively.
    The term 16.6*log10([Na+]) adjusts the Tm for changes in the salt concentration, and the term log10(0.050)
    adjusts for the salt adjustment at 50 mM Na+. Other monovalent and divalent salts will have an effect on the
    Tm of the oligonucleotide, but sodium ions are much more effective at forming salt bridges between DNA strands
    and therefore have the greatest effect in stabilizing double-stranded DNA, although trace amounts of divalent
    cations have significant and often overlooked affects.
    '''
    length = len(dna_seq)
    gc = gc_content_calc(dna_seq)

    wA = dna_seq.upper().count('A')
    yG = dna_seq.upper().count('G')
    zC = dna_seq.upper().count('C')
    xT = dna_seq.upper().count('T')
    Na = 0.05
    if len(dna_seq) < 14:
        Tm = (wA + xT) * 2 + (yG + zC) * 4 - 16.6 * 0.05 + 16.6 * math.log10(Na)
        return (round(Tm, 2), gc, length)
    if len(dna_seq) < 50:
        Tm = 100.5 + (41 * (yG + zC) / (wA + xT + yG + zC)) - (820 / (wA + xT + yG + zC)) + 16.6 * math.log10(Na)
        return (round(Tm, 2), gc, length)
    else:
        Tm = 81.5 + (41 * (yG+zC)/(wA+xT+yG+zC)) - (500/(wA+xT+yG+zC)) + 16.6*math.log10(Na)
        return (round(Tm, 2), gc, length)


def csv_to_fasta(infile: str, colnames: List[str], outfile):
    df = pd.read_csv(infile)
    my_list = df[colnames[0]].to_list()
    my_list = ['>' + k for k in my_list]
    df[colnames[0]] = my_list
    my_dict = dict(zip(df[colnames[0]], df[colnames[1]]))
    file = open(outfile, 'w')
    for i, j in enumerate(list(my_dict.keys())):
        file.write(j + '\n' + list(my_dict.values())[i] + '\n')
    file.close()


def fasta_to_df(f: str):
    l = []
    m = []
    for seq_record in SeqIO.parse(f, "fasta"):
        l.append(seq_record.id)
        m.append(str(seq_record.seq))
    return pd.DataFrame(zip(l,m))

def df_to_fasta(df: pd, cols: list, f: str):
    x = cols[0]
    y = cols[1]
    with open(f, 'w') as out:
        for i in range(df.shape[0]):
            out.write('>' + df[x].iloc[i] + '\n' + df[y].iloc[i] + '\n')

def codon_table() -> Dict[str, list]:
    return {'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
            'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
            'C': ['TGT', 'TGC'], 'W': ['TGG'],
            'E': ['GAA', 'GAG'], 'D': ['GAT', 'GAC'],
            'P': ['CCT', 'CCC', 'CCA', 'CCG'],
            'V': ['GTT', 'GTC', 'GTA', 'GTG'],
            'N': ['AAT', 'AAC'], 'M': ['ATG'],
            'K': ['AAA', 'AAG'], 'Y': ['TAT', 'TAC'],
            'I': ['ATT', 'ATC', 'ATA'], 'Q': ['CAA', 'CAG'],
            'F': ['TTT', 'TTC'], 'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
            'T': ['ACT', 'ACC', 'ACA', 'ACG'], '*': ['TAA', 'TAG', 'TGA'],
            'A': ['GCT', 'GCC', 'GCA', 'GCG'], 'G': ['GGT', 'GGC', 'GGA', 'GGG'],
            'H': ['CAT', 'CAC']}

def codon_table2() -> Dict[str, str]:
    return {
        'TCA': 'S',  # Serina
        'TCC': 'S',  # Serina
        'TCG': 'S',  # Serina
        'TCT': 'S',  # Serina
        'TTC': 'F',  # Fenilalanina
        'TTT': 'F',  # Fenilalanina
        'TTA': 'L',  # Leucina
        'TTG': 'L',  # Leucina
        'TAC': 'Y',  # Tirosina
        'TAT': 'Y',  # Tirosina
        'TAA': '*',  # Stop
        'TAG': '*',  # Stop
        'TGC': 'C',  # Cisteina
        'TGT': 'C',  # Cisteina
        'TGA': '*',  # Stop
        'TGG': 'W',  # Triptofano
        'CTA': 'L',  # Leucina
        'CTC': 'L',  # Leucina
        'CTG': 'L',  # Leucina
        'CTT': 'L',  # Leucina
        'CCA': 'P',  # Prolina
        'CCC': 'P',  # Prolina
        'CCG': 'P',  # Prolina
        'CCT': 'P',  # Prolina
        'CAC': 'H',  # Histidina
        'CAT': 'H',  # Histidina
        'CAA': 'Q',  # Glutamina
        'CAG': 'Q',  # Glutamina
        'CGA': 'R',  # Arginina
        'CGC': 'R',  # Arginina
        'CGG': 'R',  # Arginina
        'CGT': 'R',  # Arginina
        'ATA': 'I',  # Isoleucina
        'ATC': 'I',  # Isoleucina
        'ATT': 'I',  # Isoleucina
        'ATG': 'M',  # Methionina
        'ACA': 'T',  # Treonina
        'ACC': 'T',  # Treonina
        'ACG': 'T',  # Treonina
        'ACT': 'T',  # Treonina
        'AAC': 'N',  # Asparagina
        'AAT': 'N',  # Asparagina
        'AAA': 'K',  # Lisina
        'AAG': 'K',  # Lisina
        'AGC': 'S',  # Serina
        'AGT': 'S',  # Serina
        'AGA': 'R',  # Arginina
        'AGG': 'R',  # Arginina
        'GTA': 'V',  # Valina
        'GTC': 'V',  # Valina
        'GTG': 'V',  # Valina
        'GTT': 'V',  # Valina
        'GCA': 'A',  # Alanina
        'GCC': 'A',  # Alanina
        'GCG': 'A',  # Alanina
        'GCT': 'A',  # Alanina
        'GAC': 'D',  # Acido Aspartico
        'GAT': 'D',  # Acido Aspartico
        'GAA': 'E',  # Acido Glutamico
        'GAG': 'E',  # Acido Glutamico
        'GGA': 'G',  # Glicina
        'GGC': 'G',  # Glicina
        'GGG': 'G',  # Glicina
        'GGT': 'G'  # Glicina
    }

def homo_codon_usage() -> Dict[str, Tuple[str, float]]:
    return {'F': [('TTT', 0.58),('TTC', 0.42)],
            'L': [('TTA', 0.14),('TTG', 0.13), ('CTA', 0.04), ('CTC', 0.1), ('CTT', 0.12), ('CTG', 0.47)],
            'Y': [('TAT', 0.59),('TAC', 0.41)],
            '*': [('TAA', 0.61),('TAG', 0.09), ('TGA', 0.3)],
            'H': [('CAT', 0.57),('CAC', 0.43)],
            'Q': [('CAA', 0.34),('CAG', 0.66)],
            'I': [('ATT', 0.49),('ATC', 0.39),('ATA', 0.11)],
            'M': [('ATG', 1)],
            'N': [('AAT', 0.49),('AAC', 0.51)],
            'K': [('AAA', 0.74),('AAG', 0.26)],
            'V': [('GTT', 0.28),('GTC', 0.2),('GTA', 0.17),('GTG', 0.35)],
            'D': [('GAT', 0.63),('GAC', 0.37)],
            'E': [('GAA', 0.68),('GAG', 0.32)],
            'S': [('TCT', 0.17),('TCC', 0.15),('TCA', 0.14),('TCG', 0.14),('AGT', 0.16),('AGC', 0.25)],
            'C': [('TGT', 0.46),('TGC', 0.54)],
            'W': [('TGG', 1)],
            'P': [('CCT', 0.18),('CCC', 0.13),('CCA', 0.2),('CCG', 0.49)],
            'R': [('CGT', 0.36),('CGC', 0.36),('CGA', 0.07),('CGG', 0.11),('AGA', 0.07),('AGG', 0.04)],
            'T': [('ACT', 0.19),('ACC', 0.4),('ACA', 0.17),('ACG', 0.25)],
            'A': [('GCT', 0.18),('GCC', 0.26),('GCA', 0.23),('GCG', 0.33)],
            'G': [('GGT', 0.35),('GGC', 0.37),('GGA', 0.13),('GGG', 0.15)]}

def concat_pandas(l: pd):
    l = l.fillna(0)
    return ''.join((l[l.columns].astype(str).agg(''.join, axis=1)).to_list())

def amino_acids() -> dict:
    return {'Gly':'G',
            'Ala':'A',
            'Val':'V',
            'Leu':'L',
            'Ile':'I',
            'Met':'M',
            'Pro':'P',
            'Ser':'S',
            'Thr':'T',
            'Asn':'N',
            'Gln':'Q',
            'Tyr':'Y',
            'Cys':'C',
            'Trp':'W',
            'Phe':'F',
            'Lys':'K',
            'Arg':'R',
            'His':'H',
            'Asp':'D',
            'Glu':'E'}

def get_indices(element: str, l: list) -> list:
    return [i for i in range(len(l)) if element == l[i]]

def find_subset(series: list, l: list) -> int or str:
    p = get_indices(series[0], l)
    series_size = len(series)
    series_concat = ''.join(series)
    for indx in p:
        if series_concat == ''.join(l[indx:indx+series_size]):
            return indx
    return 'subset not found'

d = codon_table()

def prot_to_dna(s: str) -> str:
    '''converts protein to dna with random sampling of codon distribution'''
    output = ''
    for i in s:
        output += random.choice(d.get(i))
    return output

def gc_content_calc(s: str) -> float:
    s = s.upper()
    g = s.count('G')
    c = s.count('C')
    tot = g + c
    return round((tot/len(s))*100, 2)

def primer_design(s: str, tm: float) -> list[str]:
    s = s.upper()
    min_len = 18
    min_tm = tm-2
    max_tm = tm+2
    min_gc = 30
    max_gc = 50
    primer_list = []
    while min_len < 30:
        upper_lim = len(s) - min_len
        for i in range(0, upper_lim):
            prim = s[i:i+min_len]
            if prim[-1] == 'G' or prim[-1] == 'C':
                if oligo_calc_salt_adjust(prim) < max_tm and oligo_calc_salt_adjust(prim) > min_tm:
                    if gc_content_calc(prim) > min_gc and gc_content_calc(prim) < max_gc:
                        primer_list.append(prim)
        min_len += 1
    return primer_list
