from . import SentenceEvaluator
import logging
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from sklearn.metrics import average_precision_score
import numpy as np
from ..readers import InputExample
import torch
from torch.nn.functional import normalize
from typing import List


logger = logging.getLogger(__name__)


def get_normalized_representation(embeddings, labels):
    assert len(embeddings) == len(labels)

    embeddings_normal = embeddings[labels == 1]

    # embeddings_normal = torch.tensor(embeddings_normal, dtype=torch.float32)
    mean_pooled = torch.mean(embeddings_normal, dim=0, keepdim=True)
    normal_representation = normalize(mean_pooled, p=2, dim=1)

    return normal_representation


class BinaryClassificationEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar sentences.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.

    The labels need to be 0 for dissimilar pairs and 1 for similar pairs.

    :param sentences1: The first column of sentences
    :param sentences2: The second column of sentences
    :param labels: labels[i] is the label for the pair (sentences1[i], sentences2[i]). Must be 0 or 1
    :param name: Name for the output
    :param batch_size: Batch size used to compute embeddings
    :param show_progress_bar: If true, prints a progress bar
    :param write_csv: Write results to a CSV file
    """

    def __init__(self, sentences: List[str], labels: List[int], name: str = '', batch_size: int = 16, show_progress_bar: bool = False, write_csv: bool = True):
        self.sentences = sentences
        self.labels = labels

        assert len(self.sentences) == len(self.sentences)
        for label in labels:
            assert (label == 0 or label == 1)

        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size

        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "binary_classification_evaluation" + ("_"+name if name else '') + "_results.csv"

        self.csv_headers = ["epoch", "steps",
                            "cossim_accuracy", "cossim_accuracy_threshold", "cossim_f1", "cossim_precision",
                            "cossim_recall", "cossim_f1_threshold", "cossim_ap"]

        self.f1_threshold = 0.5
        self.normal_representation = 0

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentences = []
        scores = []

        for example in examples:
            sentences.append(example.texts[0])
            scores.append(example.label)
        return cls(sentences, scores, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info("Binary Accuracy Evaluation of the model on " + self.name + " dataset" + out_txt)

        scores = self.compute_metrices(model)

        # Main score is the max of Average Precision (AP)
        # main_score = max(scores[short_name]['ap'] for short_name in scores)
        main_score = max(scores[short_name]['accuracy'] for short_name in scores)


        file_output_data = [epoch, steps]

        for header_name in self.csv_headers:
            if '_' in header_name:
                sim_fct, metric = header_name.split("_", maxsplit=1)
                file_output_data.append(scores[sim_fct][metric])

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow(file_output_data)
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(file_output_data)

        f1_threshold = scores['cossim']['f1_threshold']
        self.f1_threshold = f1_threshold

        return main_score

    def return_f1_threshold(self):
        return self.f1_threshold

    def return_normal_representation(self):
        return self.normal_representation

    def compute_metrices(self, model):
        sentences = list(self.sentences)
        embeddings = model.encode(sentences, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
        embeddings = [emb_dict[sent] for sent in self.sentences]

        # embeddings = np.array(embeddings)
        labels = np.array(self.labels)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)

        normal_representation = get_normalized_representation(embeddings, labels)
        normal_representation_extended = normal_representation.expand(embeddings.shape[0], -1)

        # print(len(sentences))

        embeddings = np.array(embeddings)
        normal_representation_extended = np.array(normal_representation_extended)

        self.normal_representation = normal_representation

        a = paired_cosine_distances(embeddings, normal_representation_extended)
        cosine_scores = 1 - a

        output_scores = {}
        # for short_name, name, scores, reverse in [['cossim', 'Cosine-Similarity', cosine_scores, True], ['manhattan', 'Manhattan-Distance', manhattan_distances, False], ['euclidean', 'Euclidean-Distance', euclidean_distances, False], ['dot', 'Dot-Product', dot_scores, True]]:
        for short_name, name, scores, reverse in [['cossim', 'Cosine-Similarity', cosine_scores, True]]:

            acc, acc_threshold = self.find_best_acc_and_threshold(scores, labels, reverse)
            f1, precision, recall, f1_threshold = self.find_best_f1_and_threshold(scores, labels, reverse)
            ap = average_precision_score(labels, scores * (1 if reverse else -1))

            logger.info("Accuracy with {}:           {:.2f}\t(Threshold: {:.4f})".format(name, acc * 100, acc_threshold))
            logger.info("F1 with {}:                 {:.2f}\t(Threshold: {:.4f})".format(name, f1 * 100, f1_threshold))
            logger.info("Precision with {}:          {:.2f}".format(name, precision * 100))
            logger.info("Recall with {}:             {:.2f}".format(name, recall * 100))
            logger.info("Average Precision with {}:  {:.2f}\n".format(name, ap * 100))

            output_scores[short_name] = {
                'accuracy': acc,
                'accuracy_threshold': acc_threshold,
                'f1': f1,
                'f1_threshold': f1_threshold,
                'precision': precision,
                'recall': recall,
                'ap': ap
            }
        return output_scores

    @staticmethod
    def find_best_acc_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)
        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        max_acc = 0
        best_threshold = -1

        positive_so_far = 0
        remaining_negatives = sum(labels == 0)

        for i in range(len(rows)-1):
            score, label = rows[i]
            if label == 1:
                positive_so_far += 1
            else:
                remaining_negatives -= 1

            acc = (positive_so_far + remaining_negatives) / len(labels)
            if acc > max_acc:
                max_acc = acc
                best_threshold = (rows[i][0] + rows[i+1][0]) / 2

        return max_acc, best_threshold

    @staticmethod
    def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)

        scores = np.asarray(scores)
        labels = np.asarray(labels)

        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        best_f1 = best_precision = best_recall = 0
        threshold = 0
        nextract = 0
        ncorrect = 0
        total_num_duplicates = sum(labels)

        for i in range(len(rows)-1):
            score, label = rows[i]
            nextract += 1

            if label == 1:
                ncorrect += 1

            if ncorrect > 0:
                precision = ncorrect / nextract
                recall = ncorrect / total_num_duplicates
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return best_f1, best_precision, best_recall, threshold

