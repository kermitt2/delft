
import os
import shutil
import json
import time
import numpy as np
from typing import List, Union

import torch

from delft.utilities.Embeddings import Embeddings, load_resource_registry
from delft.textClassification.config import ModelConfig, TrainingConfig
from delft.textClassification.models import getModel
from delft.textClassification.preprocess import clean_text
from delft.textClassification.data_loader import create_dataloader
from delft.textClassification.trainer import Trainer

from delft import DELFT_PROJECT_DIR

class Classifier(object):
    config_file = 'config.json'
    weight_file = 'model_weights.pth'

    def __init__(self, 
                 model_name=None,
                 architecture="gru",
                 embeddings_name=None,
                 list_classes=[],
                 char_emb_size=25, 
                 dropout=0.5, 
                 recurrent_dropout=0.25,
                 use_char_feature=False, 
                 batch_size=256, 
                 optimizer='adam', 
                 learning_rate=0.001, 
                 lr_decay=0.9, 
                 clip_gradients=5.0, 
                 max_epoch=50, 
                 patience=5,
                 log_dir=None,
                 maxlen=300,
                 fold_number=1,
                 use_roc_auc=True,
                 early_stop=True,
                 class_weights=None,
                 multiprocessing=True,
                 transformer_name: str=None,
                 device=None):

        self.model_config = ModelConfig(
            model_name=model_name, 
            architecture=architecture, 
            embeddings_name=embeddings_name, 
            list_classes=list_classes, 
            char_emb_size=char_emb_size, 
            dropout=dropout, 
            recurrent_dropout=recurrent_dropout, 
            use_char_feature=use_char_feature, 
            maxlen=maxlen, 
            fold_number=fold_number, 
            batch_size=batch_size, 
            transformer_name=transformer_name)

        self.training_config = TrainingConfig(
            learning_rate=learning_rate, 
            batch_size=batch_size, 
            optimizer=optimizer, 
            lr_decay=lr_decay, 
            clip_gradients=clip_gradients, 
            max_epoch=max_epoch, 
            patience=patience, 
            use_roc_auc=use_roc_auc, 
            early_stop=early_stop, 
            class_weights=class_weights, 
            multiprocessing=multiprocessing)

        self.model = None
        self.models = None
        self.embeddings = None
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.registry = load_resource_registry(os.path.join(DELFT_PROJECT_DIR, "resources-registry.json"))

        if embeddings_name is not None:
            self.embeddings = Embeddings(embeddings_name, resource_registry=self.registry, use_ELMo=False)
            self.model_config.word_embedding_size = self.embeddings.embed_size
        else:
            self.model_config.word_embedding_size = 0

    def train(self, x_train, y_train, vocab_init=None, incremental=False, callbacks=None):
        if self.model_config.fold_number == 1:
            self.train_single(x_train, y_train, vocab_init, incremental, callbacks)
        else:
            self.train_nfold(x_train, y_train, vocab_init, incremental, callbacks)

    def train_single(self, x_train, y_train, vocab_init=None, incremental=False, callbacks=None):
        # Create Data Loaders
        # Note: We need to handle validation split here if not n-fold
        # For simplicity, let's take last 10% as validation if early_stop is True and no folds
        
        x_valid = None
        y_valid = None
        
        if self.training_config.early_stop:
            split_idx = int(len(x_train) * 0.9)
            x_valid = x_train[split_idx:]
            y_valid = y_train[split_idx:]
            x_train = x_train[:split_idx]
            y_train = y_train[:split_idx]

        # Init model
        self.model = getModel(self.model_config, self.training_config)
        self.model.to(self.device)
        
        print(f"Model: {self.model_config.architecture}")
        
        # Helper to get tokenizer if needed
        transformer_tokenizer = None
        if self.model_config.transformer_name is not None:
             # Logic to fetch tokenizer from model or transformer helper
             # In models_pytorch.py we use AutoModel. 
             # We need generic way to get tokenizer.
             from transformers import AutoTokenizer
             transformer_tokenizer = AutoTokenizer.from_pretrained(self.model_config.transformer_name)

        train_loader = create_dataloader(
            x_train, y_train, 
            self.model_config, 
            embeddings=self.embeddings,
            transformer_tokenizer=transformer_tokenizer,
            batch_size=self.training_config.batch_size,
            shuffle=True
        )
        
        valid_loader = None
        if x_valid is not None:
            valid_loader = create_dataloader(
                x_valid, y_valid, 
                self.model_config, 
                embeddings=self.embeddings,
                transformer_tokenizer=transformer_tokenizer,
                batch_size=self.training_config.batch_size,
                shuffle=False
            )

        trainer = Trainer(
            self.model,
            self.model_config,
            self.training_config,
            device=str(self.device),
            checkpoint_path=self._get_model_dir()
        )
        
        trainer.train(train_loader, valid_loader)

    def train_nfold(self, x_train, y_train, vocab_init=None, incremental=False, callbacks=None):
        pass # To implement if needed, following logic in wrapper.py

    def predict(self, texts, output_format='json', use_main_thread_only=False, batch_size=None):
        if batch_size is not None:
            self.model_config.batch_size = batch_size
            
        if self.model is None:
            raise OSError("Model not loaded")
            
        self.model.eval()
        self.model.to(self.device)
        
        transformer_tokenizer = None
        if self.model_config.transformer_name is not None:
             from transformers import AutoTokenizer
             transformer_tokenizer = AutoTokenizer.from_pretrained(self.model_config.transformer_name)

        # Preprocess texts if they are raw strings
        if len(texts) > 0 and isinstance(texts[0], str):
            # Clean text?
            # data_loader expects raw text usually and preprocesses inside Dataset if we set up logic right.
            # In TextClassificationDataset we call to_vector_single which cleans text.
            pass

        data_loader = create_dataloader(
            texts, None, 
            self.model_config,
            embeddings=self.embeddings,
            transformer_tokenizer=transformer_tokenizer,
            batch_size=self.model_config.batch_size,
            shuffle=False
        )
        
        all_preds = []
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch # no labels
                if isinstance(inputs, dict):
                     inputs = {k: v.to(self.device) for k, v in inputs.items()}
                else:
                     inputs = inputs.to(self.device)
                
                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs['logits'])
                all_preds.append(probs.cpu().numpy())
                
        result = np.concatenate(all_preds, axis=0)
        
        if output_format == 'json':
            res = {
                "software": "DeLFT",
                "date": time.ctime(),
                "model": self.model_config.model_name,
                "classifications": []
            }
            
            for i in range(len(texts)):
                classification = {
                    "text": texts[i],
                    # ... format as expected
                }
                # Simplify for now
                best_class_idx = np.argmax(result[i])
                classification["class"] = self.model_config.list_classes[best_class_idx]
                classification["score"] = float(result[i][best_class_idx])
                res["classifications"].append(classification)
            return res
        else:
            return result

    def save(self, dir_path='data/models/textClassification/'):
        directory = os.path.join(dir_path, self.model_config.model_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        self.model_config.save(os.path.join(directory, self.config_file))
        
        # Save PyTorch model
        torch.save(self.model.state_dict(), os.path.join(directory, self.weight_file))
        print(f"Model saved to {directory}")

    def load(self, dir_path='data/models/textClassification/'):
        model_path = os.path.join(dir_path, self.model_config.model_name)
        
        # Load config
        self.model_config = ModelConfig.load(os.path.join(model_path, self.config_file))
        
        # Load embeddings if needed
        if self.model_config.embeddings_name is not None:
            self.embeddings = Embeddings(self.model_config.embeddings_name, resource_registry=self.registry, use_ELMo=False)
            
        # Init model
        self.model = getModel(self.model_config, self.training_config)
        
        # Load weights
        weight_path = os.path.join(model_path, self.weight_file)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {weight_path}")

    def _get_model_dir(self):
        return os.path.join('data/models/textClassification/', self.model_config.model_name)
