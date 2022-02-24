#!/usr/bin/env python
# coding=utf-8
import torch
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss

class SentenceMatch(object):
    """
    Input two sentences, predict whether they are matched
    """
    # initialized
    def __init__(self, model_name_or_path="Fan-s/reddit-tc-bert", label_list=['matched', 'unmatched']):
        self.model_name_or_path = model_name_or_path
        self.label_list = label_list
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    # load model from huggingface hub
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
        self.model.to(self.device)

    # predict_if_match
    def __call__(self, post, response, max_seq_length=128):
        with torch.no_grad():
            args = (post, response)
            input = self.tokenizer(*args, padding="max_length", max_length=max_seq_length, truncation=True, return_tensors="pt")
            input = input.to(self.device)
            output = self.model(**input)
            logits = output.logits
            item = torch.argmax(logits, dim=1)
            predict_label = self.label_list[item]
            return predict_label


class SentencePPL(object):
    """
    input a sentence, calculate its perplexity
    """
    # initialized
    def __init__(self, model_name_or_path='microsoft/DialoGPT-medium'):
        self.model_name_or_path = model_name_or_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_func = CrossEntropyLoss(reduction='none')
        self.load_model()

    # load model from huggingface hub
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
        self.model.to(self.device)

    # calculate perplexity
    def __call__(self, text):
        input = self.tokenizer(text, return_tensors="pt")
        input = input.to(self.device)
        with torch.no_grad():
            output = self.model(**input)
            logits = output.logits
            logits = logits[:, :-1, :].contiguous()
            input = input['input_ids'][:, 1:].contiguous()
            loss_all = self.loss_func(logits.view(-1, logits.size(-1)), input.view(-1))
            loss = torch.mean(loss_all)
            ppl = torch.exp(loss)
            return ppl.item()

post = "don't make gravy with asbestos."
response = "i'd expect someone with a culinary background to know that. since we're talking about school dinner ladies, they need to learn this pronto."
text = "i really like his show, but it is really annoying when people take the show's position on things as the end all be all on the subject, and have circle jerks about it."

if __name__ == "__main__":
    sent_match = SentenceMatch()
    predict_label = sent_match.pred_if_match(post, response)[0]
    print(predict_label)
    sent_ppl = SentencePPL()
    ppl = sent_ppl.cal_ppl(text)[1]
    print(ppl)