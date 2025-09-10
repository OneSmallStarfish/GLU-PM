import argparse
from torch.utils.data import DataLoader
import logging
import json
import random

import sys
import math
import numpy as np
import torch

from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, MultiLabelAccuracyEvaluator, BinaryClassificationEvaluator

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

    file_handler = logging.FileHandler('./output/pretrain_task/FPI_hw_switch.log')

    file_handler.setFormatter(fmt)
    LOGGER.addHandler(file_handler)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("-train_data", "--train_data", type=str,
                      default="./datasets/tasks/FPI/hw_switch_train.json", help="train dataset")
    args.add_argument("-dev_data", "--dev_data", type=str,
                      default="./datasets/tasks/FPI/hw_switch_dev.json", help="dev dataset")
    args.add_argument("-test_data", "--test_data", type=str,
                      default="./datasets/tasks/FPI/hw_switch_test.json", help="test dataset")

    args.add_argument("-pretrain_model", "--pretrain_model", type=str,
                      default="roberta-base",
                      help="the path of the pretrained model to finetune")

    args.add_argument("-epoch", "--epoch", type=int,
                      default=20, help="Number of epochs")
    args.add_argument("-batch_size", "--batch_size", type=int,
                      default=16, help="Batch Size")  # 16

    args.add_argument("-outfolder", "--outfolder", type=str,
                      default="./output/task", help="Folder name to save the models.")

    args = args.parse_args()
    return args


def read_json(file):
    with open(file, 'r+') as file:
        content = file.read()
    content = json.loads(content)
    return content


def evaluate(args):
    model_save_path = args.outfolder
    train_batch_size = args.batch_size
    num_epochs = args.epoch

    train_data = read_json(args.train_data)
    dev_data = read_json(args.dev_data)
    test_data = read_json(args.test_data)

    LOGGER.info(f'model_save_path={args.outfolder}')
    LOGGER.info(f'train_batch_size={args.batch_size}')
    LOGGER.info(f'num_epochs={args.epoch}')

    LOGGER.info(f'train_data={args.train_data}')
    LOGGER.info(f'dev_data={args.dev_data}')
    LOGGER.info(f'test_data={args.test_data}')

    # load dataset
    train_samples = []
    dev_samples = []
    test_samples = []

    for item in train_data:
        train_samples.append(InputExample(texts=[item[0]], label=item[1]))
    for item in test_data:
        test_samples.append(InputExample(texts=[item[0]], label=item[1]))
    for item in dev_data:
        dev_samples.append(InputExample(texts=[item[0]], label=item[1]))

    label_count = 43
    LOGGER.info(f'label_count={label_count}')

    # load model
    model = SentenceTransformer(args.pretrain_model)
    LOGGER.info(f'pretrain_model={args.pretrain_model}')

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    dev_dataloader = DataLoader(dev_samples, shuffle=True, batch_size=train_batch_size)
    test_dataloader = DataLoader(test_samples, shuffle=True, batch_size=train_batch_size)

    # loss
    train_loss = losses.MultilabelSoftMarginLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=label_count)

    dev_evaluator = MultiLabelAccuracyEvaluator(dev_dataloader, softmax_model=train_loss, name='FPI_dev')
    test_evaluator = MultiLabelAccuracyEvaluator(test_dataloader, softmax_model=train_loss, name='FPI_test')

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    LOGGER.info("Warmup-steps: {}".format(warmup_steps))

    # finetune and evaluate
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=dev_evaluator,
              evaluator2=test_evaluator,
              epochs=num_epochs,
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              )


if __name__ == '__main__':
    init_logging(LOGGER)
    args = parse_args()
    evaluate(args)
