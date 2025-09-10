import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
from torch.utils.data import DataLoader
import logging
import json
import random

import sys
import math
import numpy as np
import torch
import pandas as pd

from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, LabelAccuracyEvaluator,BinaryClassificationEvaluator
# from sentence_transformers import get_Dataset


seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

LOGGER = logging.getLogger()


def init_logging(LOGGER):
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y-%m-%d %H:%M:%S')

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

    file_handler = logging.FileHandler('./output/pretrain/pretrain.log')
    file_handler.setFormatter(fmt)
    LOGGER.addHandler(file_handler)


def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-pretrain_data", "--pretrain_data", type=str,
                          default="./datasets/pre-train/all_log.json", help="pre-train data directory")
    args.add_argument("-abbr", "--abbr", type=str,
                      default="./datasets/pre-train/abbr.json", help="abbreviations directory")

    args.add_argument("-vocab", "--vocab", type=bool,
                      default="True", help="Whether abbreviations join the vocabulary")

    args.add_argument("-base_model", "--base_model", type=str,
                      default="roberta-base", help="base_model")

    args.add_argument("-p", "--p", type=float,
                      default=0.5, help="probability of masking abbr")
    args.add_argument("-epoch", "--epoch", type=int,
                      default=30, help="Number of epochs")
    args.add_argument("-batch_size", "--batch_size", type=int,
                      default=64, help="Batch Size")
    args.add_argument("-outfolder", "--outfolder", type=str,
                      default="./output/pretrain/pretrain/", help="Folder name to save the models.")
    args = args.parse_args()
    return args


def read_json(file):
    with open(file, 'r+') as file:
        content = file.read()
    content = json.loads(content)
    return content


def IsEnglish(character):
    for cha in character:
        if not 'A' <= cha <= 'Z':
            return False
    else:
        return True


def train(args):
    datapath = args.pretrain_data
    model_save_path = args.outfolder
    train_batch_size = args.batch_size
    num_epochs = args.epoch
    model_name = args.base_model

    LOGGER.info(f'model_save_path={args.outfolder}')
    LOGGER.info(f'train_batch_size={args.batch_size}')
    LOGGER.info(f'num_epochs={args.epoch}')
    LOGGER.info(f'pretrain_data={datapath}')
    LOGGER.info(f'base_model={args.base_model}')
    LOGGER.info(f'add_abbr={args.vocab}')

    # load data
    data = read_json(datapath)

    # load model
    word_embedding_model = models.Transformer(model_name)

    abbr_list = read_json(args.abbr)
    # add abbr to vocab
    if args.vocab:
        word_embedding_model.tokenizer.add_tokens(abbr_list, special_tokens=True)
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                    pooling_mode_cls_token=False,
                                    pooling_mode_weigth=True
                                   )

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # encode description
    desc = np.array(data)[:, 1].tolist()

    con_samples = []
    for i in range(len(data)):
        con_samples.append(InputExample(texts=[data[i][0]], embedding=desc[i]))

    random.shuffle(con_samples)

    # mask abbr
    p = args.p  # probability of masking abbr
    token_samples = []
    for i in range(len(data)):
        if (data[i][0].split('/')[0] in abbr_list and IsEnglish(data[i][0].split('/')[0])):
            if random.random() < p:
                token_samples.append(InputExample(texts=[data[i][0]], label=abbr_list.index(data[i][0].split('/')[0])))
            else:
                s = '[MASK]/' + '/'.join(data[i][0].split('/')[1:])
                token_samples.append(InputExample(texts=[s], label=abbr_list.index(data[i][0].split('/')[0])))
        elif (data[i][0].split('-')[0] in abbr_list and IsEnglish(data[i][0].split('-')[0])):
            if random.random() < p:
                token_samples.append(InputExample(texts=[data[i][0]], label=abbr_list.index(data[i][0].split('-')[0])))
            else:
                s = '[MASK]-' + '-'.join(data[i][0].split('-')[1:])
                token_samples.append(InputExample(texts=[s], label=abbr_list.index(data[i][0].split('-')[0])))

    train_con_dataloader = DataLoader(con_samples, shuffle=True, batch_size=train_batch_size)
    train_token_dataloader = DataLoader(token_samples, shuffle=True, batch_size=train_batch_size)

    # loss
    train_con_loss = losses.LogDescCLLoss(model)

    train_token_loss = losses.TokenClassificationLoss(model=model,
                                                      sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                                      num_labels=len(abbr_list))

    warmup_steps = math.ceil(len(train_con_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    model.fit(train_objectives=[(train_con_dataloader, train_con_loss), (train_token_dataloader, train_token_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            save_f=True,
    )


if __name__ == '__main__':
    init_logging(LOGGER)
    args = parse_args()
    train(args)
