# Unicode ocr

__Work in progress !__

This is an experimental optical character recognition aiming to recognize some special glyphs starting from bitmap (extracted from fonts) by mapping them to the nearest unicode codepoint.
So far, two scripts are provided:

* traindatagen.py : generates training data (png+txt) using `wordlists/wordlist_mono_clean.txt`and `wordlists/wordlist_bi_clean.txt`.
* traindata.py : train the model from training data.