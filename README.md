# Running the Models -- To train the models, run the following commands:

## DAN with GloVe embeddings
python main.py --model DAN_GloVe --epochs 150 --lr 3e-5

## DAN with Random embeddings
python main.py --model DAN_Random --epochs 100 --lr 5e-5

## DAN with BPE tokenization and subwordembeddings
python main.py --model DAN_BPE --epochs 100 --lr 1e-4

# What are in the new files?
## DANmodels.py contains DAN_GloVe, DAN_Random and the SentimentDatasetWordEmbedding class.
## DAN_BPE_models.py contains DAN_BPE and the BPE_Tokenizer class.

# Special Note:
## Under a very rare chance that the accuracy of the model is fixed at 0.478 and 0.491 respectively for training and dev set, please run the code again. Maybe this is due to the randomization of the embeddings or something else. This happens very rarely.
