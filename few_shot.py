from audioop import mul
import torchaudio
import urllib3
from easyfsl.datasets import CUB
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Omniglot
from torchvision.models import resnet18
from tqdm import tqdm
from easyfsl.utils import plot_images, sliding_average
from easyfsl.samplers import TaskSampler
from pathlib import Path
import random
from statistics import mean
import numpy as np
# from easyfsl.methods import PrototypicalNetworks, FewShotClassifier
from easyfsl.modules import resnet12
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from easyfsl.methods.utils import evaluate
from torch.utils.data import Dataset
import pandas as pd
from typing import Any, Callable, List, Optional, Tuple
from sklearn import metrics
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
# import librosa


class Material(Dataset):

    def __init__(self,  annotations_file, transformation,
                 target_sample_rate, num_samples):
        self.annotations = pd.read_csv(annotations_file)
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

        self._flat_character_images = list(
            zip(self.annotations.iloc[:, 0], self.annotations.iloc[:, 1]))

        self._characters = self.annotations.iloc[:, 0]
        # self.labels: List[Tuple[str, int]] = sum(self._character_images, [])
        # self.data = self.annotations.iloc[:, 0]

        # self.split = "train" if training else "test"

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)

        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)

        multi_channel_signal = torch.zeros([3, 64, 29])
        multi_channel_signal[0, :, :] = signal[0, :, :]
        multi_channel_signal[1, :, :] = signal[0, :, :]
        multi_channel_signal[2, :, :] = signal[0, :, :]
        signal = multi_channel_signal
        return signal, label

    def get_labels(self):
        labels = [instance[1] for instance in self._flat_character_images]
        return labels

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        # fold = f"fold{self.annotations.iloc[index, 5]}"
        path = self.annotations.iloc[index, 0]
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 1]


class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        return scores


def evaluate_on_one_task(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> [int, int]:
    """
    Returns the number of correct predictions of query labels, and the total number of predictions.
    """

    # model.eval()
    # example_scores = model(
    #     example_support_images,
    #     example_support_labels,
    #     example_query_images,
    # ).detach()

    # _, example_predicted_labels = torch.max(model(
    #     example_support_images,
    #     example_support_labels,
    #     example_query_images,
    # ).detach().data, 1)

    return (
        torch.max(
            model(support_images,
                  support_labels, query_images)
            .detach()
            .data,
            1,
        )[1]
        == query_labels
    ).sum().item(), len(query_labels), torch.max(
        model(support_images,
              support_labels, query_images)
        .detach()
        .data,
        1,
    )[1]


def evaluate(data_loader: DataLoader):
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
    model.eval()
    i = 0
    predicted = []
    actual = []
    with torch.no_grad():
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
        ) in tqdm(enumerate(data_loader), total=len(data_loader)):

            correct, total, predicted_labels = evaluate_on_one_task(
                support_images, support_labels, query_images, query_labels
            )

            total_predictions += total
            correct_predictions += correct

            if(i < 6):
                actual.append(class_ids[query_labels[i]])
                predicted.append(
                    class_ids[predicted_labels[i]])
            else:
                i = 0
                actual.append(class_ids[query_labels[i]])
                predicted.append(
                    class_ids[predicted_labels[i]])

            i += 1

        for i in range(len(actual)):
            if(actual[i] == 1):
                print()
                actual[i] = 'Cardboard'
            elif(actual[i] == 2):
                actual[i] = 'Metal'
            elif(actual[i] == 3):
                actual[i] = 'Plastic'

        for i in range(len(predicted)):
            if(predicted[i] == 1):
                predicted[i] = 'Cardboard'
            elif(predicted[i] == 2):
                predicted[i] = 'Metal'
            elif(predicted[i] == 3):
                predicted[i] = 'Plastic'

        confusion_matrix = metrics.confusion_matrix(
            actual, predicted, labels=['Cardboard', 'Metal', 'Plastic'])

        cm_display = metrics.ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix, display_labels=['Cardboard', 'Metal', 'Plastic'])

        cm_display.plot()
        plt.show()

        print('\nClassification Report\n')
        print(classification_report(actual, predicted,
              target_names=['Cardboard', 'Metal', 'Plastic']))

    # print("Ground Truth / Predicted")
    # print(len(query_labels))
    # for i in range(len(query_labels)):
    #     print(
    #         f"{test_set._characters[example_class_ids[query_labels[i]]]} / {test_set._characters[example_class_ids[predicted_labels[i]]]}"
    #     )
    print(
        f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions/total_predictions):.2f}%")
    return (100 * correct_predictions/total_predictions)


def fit(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> float:
    optimizer.zero_grad()
    classification_scores = model(
        support_images, support_labels, query_images
    )

    loss = criterion(classification_scores, query_labels)
    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == '__main__':
    image_size = 28

    convolutional_network = resnet18(pretrained=True)
    convolutional_network.fc = nn.Flatten()
    # print(convolutional_network)

    model = PrototypicalNetworks(convolutional_network)

    ANNOTATIONS_FILE_TRAIN = "/Users/kennedydike/Desktop/University/Masters/Thesis/test folder/audio_dataset_train.csv"
    NUM_SAMPLES = 14399
    SAMPLE_RATE = 48000

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    # Disable SSL warnings that may happen during download.

    # n_test_tasks = 1000

    # TRAINING SET HYPER PARAMETERS DEFAULT
    # N_WAY = 3  # Number of classes in a task
    # N_SHOT = 5  # Number of images per class in the support set
    # N_QUERY = 10  # Number of images per class in the query set
    # N_EVALUATION_TASKS = 100
    # N_TRAINING_EPISODES = 20
    # N_VALIDATION_TASKS = 20
    # LR = 0.001
    # log_update_frequency = 10

    N_WAY = 3  # Number of classes in a task  #3
    N_SHOT = 10  # Number of images per class in the support set #10
    N_QUERY = 5  # Number of images per class in the query set #5
    N_EVALUATION_TASKS = 200  # 100
    N_TRAINING_EPISODES = 50  # 50
    LR = 0.00001  # 0.001
    log_update_frequency = 50  # 50

    # N_VALIDATION_TASKS = 20

    train_set = Omniglot(
        root="./data",
        background=True,
        transform=transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        download=True,
    )

    train_sampler = TaskSampler(
        train_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_TRAINING_EPISODES
    )
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Train the model yourself with this cell

    # all_loss = []
    # model.train()
    # with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
    #     for episode_index, (
    #         support_images,
    #         support_labels,
    #         query_images,
    #         query_labels,
    #         _,
    #     ) in tqdm_train:
    #         loss_value = fit(support_images, support_labels,
    #                          query_images, query_labels)
    #         all_loss.append(loss_value)

    #         if episode_index % log_update_frequency == 0:
    #             tqdm_train.set_postfix(loss=sliding_average(
    #                 all_loss, log_update_frequency))

    # torch.save(model.state_dict(), 'trained_few_shot_model')

    model = PrototypicalNetworks(convolutional_network)
    model.load_state_dict(torch.load('trained_few_shot_model'))
    print('Model Loaded')
    # model.eval()

    urllib3.disable_warnings()
    ANNOTATIONS_FILE_TEST = "/Users/kennedydike/Desktop/University/Masters/Thesis/test folder/audio_dataset.csv"
    test_set = Material(ANNOTATIONS_FILE_TEST, transformation=mel_spectrogram,
                        target_sample_rate=SAMPLE_RATE, num_samples=NUM_SAMPLES)

    # TESTING SET HYPER PARAMETERS DEFAULT
    # N_WAY = 3  # Number of classes in a task
    # N_SHOT = 3  # Number of images per class in the support set
    # N_QUERY = 2  # Number of images per class in the query set
    # N_EVALUATION_TASKS = 100

    accuracy_analysis = []
    num_loop = []
    for shot in range(6, 7):
        N_WAY = 3  # Number of classes in a task
        N_SHOT = shot  # Number of images per class in the support set
        N_QUERY = 2  # Number of images per class in the query set
        N_EVALUATION_TASKS = 100

        test_sampler = TaskSampler(
            test_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
        )

        test_loader = DataLoader(
            test_set,
            batch_sampler=test_sampler,
            num_workers=8,
            pin_memory=True,
            collate_fn=test_sampler.episodic_collate_fn,
        )

        (
            example_support_images,
            example_support_labels,
            example_query_images,
            example_query_labels,
            example_class_ids,
        ) = next(iter(test_loader))

        # print(evaluate(test_loader))
        # accuracy_analysis.append(evaluate(test_loader))
        # num_loop.append(shot)
        evaluate(test_loader)
    # plt.plot(num_loop, accuracy_analysis)
    # plt.xlabel("Number of shots")
    # plt.ylabel("Accuracy of Classification")
    # plt.title("N-shot Anaylsis")
    # plt.show()
