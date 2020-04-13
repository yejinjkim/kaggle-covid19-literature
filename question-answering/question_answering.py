import pandas as pd
import torch
from transformers import BertTokenizer
from transformers import BertForQuestionAnswering


def load_pretrained_qa_model(model_str=None, use_cuda=True):
    if model_str is None:
        model_str = 'bert-large-uncased-whole-word-masking-finetuned-squad'
        device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    
    tokenizer = BertTokenizer.from_pretrained(model_str)
    model = BertForQuestionAnswering.from_pretrained(model_str).to(device)

    model.eval()
    return tokenizer, model

def answer_question(question, document, model, tokenizer):
    device = model.device
    
    encoded = tokenizer.encode_plus(question, document, return_tensors='pt', max_length=512)
    start_scores, end_scores = model(encoded['input_ids'].to(device),
                                     token_type_ids=encoded['token_type_ids'].to(device))

    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'].squeeze())
    ans_start, ans_end = torch.argmax(start_scores), torch.argmax(end_scores)
    
    ans_tokens = tokens[ans_start: ans_end+1]
    if '[SEP]' in ans_tokens:
        ans_tokens = ans_tokens[ans_tokens.index('[SEP]')+1:]
    ans = tokenizer.convert_tokens_to_string(ans_tokens)
    ans = ans.replace(' - ', '-').replace('[CLS]', '')

    return ans