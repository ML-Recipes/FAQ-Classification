from transformers import BertForSequenceClassification, BertTokenizerFast
from preprocessing import Preprocessing
from shared.utils import get_doc_length
from shared.utils import load_json
from functools import lru_cache
import torch
import torch.nn.functional as F
import logging
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FAQClassifier(object):
    """ Class for predicting label in new documents
    
    :param top_k: top k predictions
    :param lang_code: language model
    :param min_words: min number of words for prediction
    :param min_conf_score: minimum confidence threshold
    :param max_length: max sequence length required by BERT pre-trained model
    :param pre_trained_name: pre-trained model name
    """
    def __init__(self, lang_code='en', min_words=1, min_conf_score=0.10, max_length=256, top_k=1, pre_trained_name='bert-base-uncased'):
        
        self.top_k = top_k
        self.lang_code = lang_code
        self.min_words = min_words
        self.min_conf_score = min_conf_score
        self.max_length = max_length
            
        self.model_path = "models/" + self.lang_code + "/model.pt"
        self.label_path = "labels/" + self.lang_code + "/labels.json"

        if os.path.isfile(self.model_path) and os.path.isfile(self.label_path):
            self.labels = load_json(self.label_path)
            self.tokenizer = self.load_tokenizer(pre_trained_name)
            self.model = self.load_model(pre_trained_name, num_labels=len(self.labels))
            self.model.to(device)
            self.model.load_state_dict(torch.load(self.model_path, device), strict=False)

    
    @lru_cache(maxsize=128)
    def load_model(self, pre_trained_name, num_labels):
        """ Load BERT pre-trained model with given num labels 
        
        :param pre_trained_name: BERT pre-trained model name
        :param num_labels: total number of classes
        :return: BertForSequenceClassification model
        """
        model = BertForSequenceClassification.from_pretrained(
            pre_trained_name, 
            num_labels=num_labels,
            output_attentions=False, 
            output_hidden_states=False
        )
        return model

    @lru_cache(maxsize=128)
    def load_tokenizer(self, pre_trained_name):
        """ Load BERT pre-trained tokenizer
        
        :param pre_trained_name: BERT pre-trained model name
        :return: BERT tokenizer
        """
        tokenizer = BertTokenizerFast.from_pretrained(pre_trained_name)
        return tokenizer 
    
    def predict(self, text):
        """ Predict label for new documents 
        
        :param text: text 
        :return: python dictionary
        """
        try:
            prediction = dict()

            if text:
                if get_doc_length(text) > self.min_words:
                    if os.path.isfile(self.model_path) and os.path.isfile(self.label_path):
                        # tokenize, encode and generate input_ids, attention_mask
                        p = Preprocessing(self.tokenizer)
                        input_ids, attention_mask = p.text_preprocessing([text], max_length=self.max_length)
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        
                        # generate index_label dictionary and get the list of class names
                        index_label = {v: k for k, v in self.labels.items()}
                        class_names = list(index_label.values())

                        predictions = []
                        with torch.no_grad():
                            # add input_ids, attention_mask to BERT pre-trained model
                            outputs = self.model(input_ids, attention_mask)                     
                            tensors = outputs[0][0]                                             
                            top_k_preds = torch.topk(tensors, self.top_k)                   
                            tensor_scores = top_k_preds[0]                                      
                            tensor_indexes = top_k_preds[1]                                                     
                            # convert tensors to probabilities
                            confidences = F.softmax(tensor_scores, dim=0)                       
                            confidences = confidences.tolist()                                  
                        
                            # loop through each confidence and get associated index label
                            for index, confidence in enumerate(confidences):
                                # get confidence tensor position
                                tensor_index = top_k_preds[1][index].item() 
                                label_pred = dict()
                                label_pred['label'] = class_names[tensor_index]
                                label_pred['confidence'] = "{0:.4f}".format(confidence)
                                predictions.append(label_pred)
                        
                        if predictions:
                            max_conf_label = max(predictions, key=lambda k: k["confidence"])
                            label = max_conf_label.get("label")
                            confidence = max_conf_label.get("confidence")
                            
                            if float(confidence) <= self.min_conf_score:
                                return "unknown label, confidence below threshold"
                    
                            prediction["label"] = label
                            prediction["confidence"] = confidence
                            prediction["predictions"] = predictions
                            prediction["message"] = "successful"
                            return prediction
                        else:
                            return "no labels found"
                    else:
                        return "model not found"
                else:
                    return "required at least {} words for prediction".format(self.min_words)
            else:
                return "required textual content"
        except Exception:
            logging.error("exception occured", exc_info=True)



