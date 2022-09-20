import sys
import os
import numpy as np 
from delft.sequenceLabelling.reader import load_data_and_labels_conll
from delft.utilities.Utilities import truecase_sentence

"""
Convert CoNLL tokens casing via truecase
"""

def convert(input_file, output_file):
    with open(output_file, "w") as outfile:
        with open(input_file, "r+", encoding="UTF-8") as infile:
            words, tags = [], []
            for original_line in infile:
                line = original_line.rstrip()
                if len(line) == 0 or line.startswith('-DOCSTART-') or line.startswith('#begin document'):
                    if len(words) != 0:
                        words = truecase_sentence(words)
                        for i in range(len(words)):
                            outfile.write(words[i])
                            outfile.write("\t")  
                            outfile.write(tags[i])
                            outfile.write("\n")  
                        words, tags = [], []
                    outfile.write(original_line)
                else:
                    if len(line.split('\t')) == 2:
                        word, tag = line.split('\t')
                    else:
                        word, _, tag = line.split('\t')
                    words.append(word)
                    tags.append(tag)

if __name__ == "__main__":
    convert("../../data/sequenceLabelling/CoNLL-2003/eng.testa", "../../data/sequenceLabelling/CoNLL-2003-TC/eng.testa")
    convert("../../data/sequenceLabelling/CoNLL-2003/eng.testb", "../../data/sequenceLabelling/CoNLL-2003-TC/eng.testb")
    convert("../../data/sequenceLabelling/CoNLL-2003/eng.train", "../../data/sequenceLabelling/CoNLL-2003-TC/eng.train")
