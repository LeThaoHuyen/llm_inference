# credit: https://github.com/kojima-takeshi188/zero_shot_cot
from transformers import LlamaForCausalLM, LlamaTokenizer
import pandas as pd
import random
import argparse
import logging
import numpy as np
import multiprocessing
import torch
from utils import *
from dataset import setup_data_loader
from tqdm.auto import tqdm

class Decoder:
    def __init__(self):
        model_name = "meta-llama/Llama-2-7b-hf"
        self.device = torch.device("cuda")
        self.model = LlamaForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)

    def decode(self, input_text, max_length):
        input_ids = self.tokenizer(input_text, return_tensors="pt")["input_ids"].to(self.device)
        generated_ids = self.model.generate(input_ids, max_length=max_length)
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # response = generated_ids
        return response[0]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument("--data_start_idx", type=int, default=0, help="first data sample idx to take")
    parser.add_argument("--data_end_idx", type=int, default=-1, help="last data sample idx to take, -1 means taking all samples")
    parser.add_argument("--result_path", type=str, help="path to file for saving generated answers (file in csv format)")

    parser.add_argument("--random_seed", type=int, default=3010, help="random seed")
    
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",  "strategyqa", "svamp", "singleeq", "bigbench_date", "object_tracking", "coin_flip", "last_letters"], help="dataset used for experiment"
    )
    
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")
    
    parser.add_argument("--max_num_worker", type=int, default=3, help="maximum number of workers for dataloader")
    
    # parser.add_argument(
    #     "--model", type=str, default="gpt3", choices=["gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl"], help="model used for decoding. Note that 'gpt3' are the smallest models."
    # )
    
    parser.add_argument(
        "--method", type=str, default="few_shot_cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot"], help="method"
    )
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1, help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=930, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    # parser.add_argument(
    #     "--api_time_interval", type=float, default=1.0, help=""
    # )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "/cm/shared/huyenlt44/GoT/data/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    
    args.direct_answer_trigger_for_fewshot = "The answer is"
    
    if args.cot_trigger_no == 1:
        args.cot_trigger = "Let's think step by step."
    elif args.cot_trigger_no == 2:
        args.cot_trigger = "We should think about this step by step."
    elif args.cot_trigger_no == 3:
        args.cot_trigger = "First,"
    elif args.cot_trigger_no == 4:
        args.cot_trigger = "Before we dive into the answer,"
    elif args.cot_trigger_no == 5:
        args.cot_trigger = "Proof followed by the answer."
    elif args.cot_trigger_no == 6:
        args.cot_trigger = "Let's think step by step in a realistic way."
    elif args.cot_trigger_no == 7:
        args.cot_trigger = "Let's think step by step using common sense and knowledge."
    elif args.cot_trigger_no == 8:
        args.cot_trigger = "Let's think like a detective step by step."
    elif args.cot_trigger_no == 9:
        args.cot_trigger = "Let's think about this logically."
    elif args.cot_trigger_no == 10:
        args.cot_trigger = "Let's think step by step. First,"
    elif args.cot_trigger_no == 11:
        args.cot_trigger = "Let's think"
    elif args.cot_trigger_no == 12:
        args.cot_trigger = "Let's solve this problem by splitting it into steps."
    elif args.cot_trigger_no == 13:
        args.cot_trigger = "The answer is after the proof."
    elif args.cot_trigger_no == 14:
        args.cot_trigger = "Let's be realistic and think step by step."
    else:
        raise ValueError("cot_trigger_no is not properly defined ...")
    
    return args
    
def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    
    fix_seed(args.random_seed)
    
    # load model
    print("load model...")
    decoder = Decoder()
    
    # load data
    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    
    if args.method == "few_shot":
        demo = create_demo_text(cot_flag=False, direct_answer_trigger_for_fewshot=args.direct_answer_trigger_for_fewshot)
    elif args.method == "few_shot_cot":
        demo = create_demo_text(cot_flag=True, direct_answer_trigger_for_fewshot=args.direct_answer_trigger_for_fewshot)
    else:
        pass

    total = 0
    correct_list = [] 
    questions = []
    answers = []      
    model_answers = [] 
    extracted_model_answers = []

    pbar = tqdm(range(len(dataloader)))
    for i, data in enumerate(dataloader):
        print('*************************')
        print("{}st data".format(i+1))
                
        # Prepare question template ...
        x, y = data
        x = "Q: " + x[0] + "\n" + "A:"
        y = y[0].strip()
        
        questions.append(x)
        answers.append(y)

        if args.method == "zero_shot":
            x = x + " " + args.direct_answer_trigger_for_zeroshot
        elif args.method == "zero_shot_cot":
            x = x + " " + args.cot_trigger
        elif args.method == "few_shot":
            x = demo + x
        elif args.method == "few_shot_cot":
            # x = demo + "Please answer this question: \n" + x
            x = demo + x
        else:
            raise ValueError("method is not properly defined ...")
        # print(x)
        # return 

        # Answer prediction by generating text ...
        max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
        # z = decoder.decode(args, x, max_length, i, 1)
        # print(x) # to-remove
        # print(y) # to-remove
        z = decoder.decode(x, max_length)

        print("Max length:", max_length)

        # Answer extraction for zero-shot-cot ...
        if args.method == "zero_shot_cot":
            z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
            max_length = args.max_length_direct
            # pred = decoder.decode(args, z2, max_length, i, 2)
            pred = decoder.decode(z2, max_length)
            # print(z2 + pred)
        else:
            pred = z
            # print(x + pred)

        # save pred
        model_answers.append(pred)

        # Clensing of predicted answer ...
        pred = answer_cleansing(args, pred)
        extracted_model_answers.append(pred)
        
        # Choose the most frequent answer from the list ...
        print("pred : {}".format(pred))
        print("GT : " + y)
        print('*************************')
        
        pbar.update(1)

        # Checking answer ...
        correct = (np.array([pred]) == np.array([y])).sum().item()
        correct_list.append(correct)
        total += 1 #np.array([y]).size(0)
        
        if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
            break
            #raise ValueError("Stop !!")
        
        if i%10 == 0:
            tmp = pd.DataFrame({
                "question": questions,
                "answer": answers,
                "model_answer": model_answers,
                "extracted_model_answer": extracted_model_answers,
            })
            tmp.to_csv("temp.csv")
    
    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("accuracy : {}".format(accuracy))

    # Save all result
    results = pd.DataFrame({
        "question": questions,
        "answer": answers,
        "model_answer": model_answers,
        "extracted_model_answer": extracted_model_answers,
    })

    results.to_csv(args.result_path)
    print("save results successfully!")

if __name__=="__main__":
    main()

    