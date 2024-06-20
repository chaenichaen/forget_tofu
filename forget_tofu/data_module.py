import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
from utils import get_model_identifiers_from_yaml, add_dataset_index
import json


def gpt4_convert_raw_data_to_model_format(tokenizer, max_length, paraphrase_question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs["question_start_tag"], model_configs["question_end_tag"], model_configs["answer_tag"]
    paraphrase_question = question_start_token + paraphrase_question + question_end_token
    new_answer = answer_token + answer
    full_text = paraphrase_question + new_answer
    num_paraphrase_question_tokens = len(tokenizer.tokenize(paraphrase_question,add_special_tokens=True))
    
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True
    )
    #패딩 토큰(Pad Token): 패딩 토큰은 일반적으로 여러 문장 또는 문서를 동일한 길이로 만들기 위해 사용됩니다. 예를 들어, 배치 처리를 위해 여러 문장을 동일한 길이로 만들어야 하는 경우, 짧은 문장에 패딩 토큰을 추가하여 모든 문장의 길이를 동일하게 만들 수 있습니다. 패딩 토큰은 보통 0 또는 특정한 값을 가지며, 모델 학습에 직접적인 영향을 주지 않습니다
    #종료 토큰(End of Sentence, EOS Token): 종료 토큰은 문장의 끝을 나타내는 토큰입니다. 이 토큰은 모델이 문장의 끝을 인식하게 도와줍니다. 특히, 문장 생성 작업(예: 기계 번역, 텍스트 생성 등)에서는 모델이 생성된 문장을 언제 끝내야 하는지 결정하는 데 종료 토큰이 사용됩니다.
    pad_length = max_length - len(encoded['input_ids'])
    #'input_ids': [101, 7592, 1010, 2088, 999, 102]
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id]*pad_length
    pad_attention_mask = encoded['attention_mask'] + [0]*pad_length
    
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded.input_ids + [-100]*pad_length
        #-100을 주는이유: 레이블 값이 -100인 토큰은 무시하도록 

    for i in range(num_paraphrase_question_tokens):
        label[i] = -100

    return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)

def convert_json_to_list(data):
    question_list = [item["question"] for item in data]
    return question_list


def convert_raw_data_to_model_format(tokenizer, max_length, question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs["question_start_tag"], model_configs["question_end_tag"], model_configs["answer_tag"]
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question,add_special_tokens=True))
    
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True
    )
    #패딩 토큰(Pad Token): 패딩 토큰은 일반적으로 여러 문장 또는 문서를 동일한 길이로 만들기 위해 사용됩니다. 예를 들어, 배치 처리를 위해 여러 문장을 동일한 길이로 만들어야 하는 경우, 짧은 문장에 패딩 토큰을 추가하여 모든 문장의 길이를 동일하게 만들 수 있습니다. 패딩 토큰은 보통 0 또는 특정한 값을 가지며, 모델 학습에 직접적인 영향을 주지 않습니다
    #종료 토큰(End of Sentence, EOS Token): 종료 토큰은 문장의 끝을 나타내는 토큰입니다. 이 토큰은 모델이 문장의 끝을 인식하게 도와줍니다. 특히, 문장 생성 작업(예: 기계 번역, 텍스트 생성 등)에서는 모델이 생성된 문장을 언제 끝내야 하는지 결정하는 데 종료 토큰이 사용됩니다.
    pad_length = max_length - len(encoded['input_ids'])
    #'input_ids': [101, 7592, 1010, 2088, 999, 102]
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id]*pad_length
    pad_attention_mask = encoded['attention_mask'] + [0]*pad_length

    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded.input_ids + [-100]*pad_length
        #-100을 주는이유: 레이블 값이 -100인 토큰은 무시하도록 
    
    #change label to -100 for question tokens
    for i in range(num_question_tokens):
        label[i] = -100
    return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)


def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)
    

def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)


import json
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class Add_forget_TextDataset(Dataset):
    def __init__(self, data_path, arg_question, tokenizer, model_family, max_length=512, split=None, question_key='question', answer_key='answer'):
        super(Add_forget_TextDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_dataset(data_path, split)["train"]

        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.ak = answer_key
        with open(arg_question, 'r', encoding='utf-8') as file:
            input_data = json.load(file)
            self.paraphrase_questions = convert_json_to_list(input_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        answers = self.data[idx][self.ak]
        paraphrase_questions = self.paraphrase_questions[idx]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = gpt4_convert_raw_data_to_model_format(
                self.tokenizer, self.max_length, paraphrase_questions, answer, self.model_configs
            )
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return torch.stack(pad_input_ids_list).squeeze(), \
               torch.stack(label_list).squeeze(), \
               torch.stack(pad_attention_mask_list).squeeze(), \
               torch.tensor(indices)

  
class TextForgetDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family,  max_length=512, split = "forget10", loss_type="idk"):
        super(TextForgetDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = datasets.load_dataset(data_path, split)["train"]
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        self.retain_data =datasets.load_dataset(data_path, retain_split)["train"]
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

        if self.loss_type == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idontknowfile = "data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
        else:
            self.split1, self.split2 = "forget", "retain"

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        for data_type in [self.split1, self.split2]:
            #use questions from forget set if split is idk or forget
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            question = data[idx]['question']
            answer = data[idx]['answer']

            if data_type == "idk":
                #get a random answer position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()
                
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets


class TextForgetDatasetDPOQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = "forget10", ):
        super(TextForgetDatasetDPOQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = datasets.load_dataset(data_path, split)["train"]
        self.idontknowfile = "data/idontknow.jsonl"
        self.idk = open(self.idontknowfile, "r").readlines()
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in ["idk", "forget", "retain"]:
            data = self.forget_data if data_type != "retain" else self.retain_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            
            question = data[idx]['question']
            
            if data_type != "idk":
                answer = data[idx]['answer']
            else:
                #get a random position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets


class TextDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None, question_key='question', answer_key='answer'):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # data_len = len(datasets.load_dataset(data_path, split)["train"])
        # self.data = datasets.load_dataset(data_path, split)["train"].select(range(min(100, data_len)))
        self.data = datasets.load_dataset(data_path, split)["train"]

        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)


def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)

def custom_data_collator_with_indices(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    indices = [s[3] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(indices)

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss