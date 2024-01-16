Grobid Header model
===================

These figures are obtained with the end-to-end evaluation using the PMC evaluation set of 1943 documents. The header models were trained on all available training data. Training and end-to-end evaluation were performed from Grobid command line. Runtimes are with GPU (nvidia 1080Ti) and 4 core CPU (i7-7700K CPU @4.20GHz).

| architecture		| base model  | F1-score   | runtime(s) |   
|---     		      |---          |---         |---    |
| BERT_CRF_FEATURES	| scibert     | 78.85	   | 1537  |   
| BERT_ChainCRF_FEATURES| scibert     | 85.75	   |  590  |   
| BERT_CRF		      | scibert     | 78.61 	   | 1120  |   
| BERT_ChainCRF		| scibert     | 85.73	   |  492  |   
| BERT_FEATURES		| scibert     | 85.59	   |  564  |   
| BERT			| scibert     | 85.77	   |  441  |   
| -                     |             | 	         |       |   
| BidLSTM_CRF_FEATURES	| scibert     | 90.97	   |  838  |   
| BidLSTM_ChainCRF_FEATURES|scibert   | 91.02	   |  681  |   
| BidLSTM_CRF		| scibert     | 87.27	   |  805  |   
| BidLSTM_ChainCRF	| scibert     | 87.98	   |  668  |   
| -    			|             | 	         |       |   
| BERT			| bert        | 79.83	   |  433  |   
| BERT_FEATURES		| bert        | 79.44	   |  570  |   
| -                     |             | 	         |       |   
| CRF (Wapiti)          | -           | 91.1	   |  426  |   

Reported F1-score are averaged micro F1-score considering Ratcliff/Obershelp Matching (Minimum Ratcliff/Obershelp similarity at 0.95 or more)

**Note 1:** 

scibert = allenai/scibert_scivocab_cased
bert = bert-base-cased
      
**Note2:**

incremental training increase slightly but systematically RNN-based architecture ( around +0.2 to +0.3)
incremental training tends to decreate BERT-based architecture (usually -1 to -1.5)

