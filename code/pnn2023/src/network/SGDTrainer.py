import random
import time
import numpy as np
from tqdm.auto import tqdm
from typing import List

from src.network.Network import Network
from src.network.UpdateMechanism import UpdateMechanism


class SGDTrainer:
    batchSize: int
    learningRate: float
    amountEpochs: int
    shuffle: bool
    updateMechanism: UpdateMechanism
    debug: bool
    loss: bool
    loss_batch: int
    dev: bool
    dev_improv: int

    def __init__(self, learningRate=0.01, amountEpochs=1, debug=False, loss=True, loss_batch=100, dev=False, dev_improv=1):
        self.batchSize = 1
        self.learningRate = learningRate
        self.amountEpochs = amountEpochs
        self.shuffle = True
        self.updateMechanism = UpdateMechanism.StandardSGD
        self.debug = debug
        self.loss = loss
        self.loss_batch = loss_batch
        self.dev = dev
        # 1 == .01 --> right shifted
        self.dev_improv = dev_improv

    def optimize(self, net: Network, train: [(List[any], [float])], dev: [(List[any], [float])], eval: [(List[any], [float])]):
        if self.loss: self.debug = False
        net.debug = self.debug

        total_samples = 0
        total_loss = 0
        batch_correct = 0
        prev_acc = -1.0
        for e in range(self.amountEpochs):
            time.sleep(0.5)
            with tqdm(total=len(train), disable=not self.loss, leave=True, position=0) as pbar:
                if self.debug:
                    print('** Epoch {0} **'.format(e+1))
                    print()
                    print()

                # due to batch size 1, else we'd need bn
                for index, sample in enumerate(train):
                    if self.debug:
                        print('** Iteration {0} **'.format(index+1))
                        print()

                    # assuming data is split in a tuple of [float] input and [float] output
                    input = sample[0]
                    labels = sample[1]

                    # loss
                    loss = net.forward(input, labels)[0].elements
                    total_loss += loss
                    total_samples += 1

                    # accuracy
                    softmax = net.cache[-2]
                    pred = np.argmax(softmax[0].elements.flatten())
                    label = np.argmax(labels)
                    if pred == label: batch_correct += 1

                    if self.loss and index % self.loss_batch == 0:
                        pbar.set_description(f'TRAINING Phase - Epoch {e+1} '
                                             f'| Current average loss: {(total_loss / total_samples):.4f} '
                                             f'| Batch accuracy: {(batch_correct / self.loss_batch):.4f} '
                                             f'| Measured in intervals of {self.loss_batch}')
                        pbar.update(self.loss_batch)
                        batch_correct = 0

                    net.backprop(self.learningRate)

                    if self.debug:
                        print()
                        print()

                if self.shuffle:
                    random.shuffle(train)

                # early break when improvement from last time is too low
                dev_acc = self.evaluate(dev, net, batch=self.loss_batch, state='DEV', e=e)
                if self.loss: print(f"Dev accuracy: {dev_acc:.4f}")
                if self.dev and \
                   (int(round(dev_acc, 2) * 100) - int(round(prev_acc, 2) * 100)) < self.dev_improv:
                    eval_acc = self.evaluate(eval, net, batch=self.loss_batch, state='EVAL', e=e)
                    return eval_acc
                prev_acc = dev_acc

                if self.debug:
                    print()
                    print()
                    print()

        eval_acc = self.evaluate(eval, net, batch=self.loss_batch, state='EVAL', e=self.amountEpochs-1)
        return eval_acc

    def evaluate(self, dataset, net, batch=100, state='TRAINING', e=0):
        time.sleep(0.5)

        correct = 0
        batch_correct = 0
        total_loss = 0
        total_samples = 0
        with tqdm(total=len(dataset), disable=not self.loss, leave=True, position=0) as pbar_eval:
            for index, (image, label) in enumerate(dataset):
                # loss
                loss = net.forward(image, label)[0].elements
                total_loss += loss
                total_samples += 1

                # accuracy
                softmax = net.cache[-2]
                pred = np.argmax(softmax[0].elements.flatten())
                label = np.argmax(label)
                if pred == label:
                    correct += 1
                    batch_correct += 1

                if index % batch == 0:
                    pbar_eval.set_description(f'{state} Phase - Epoch {e+1} '
                                              f'| Current average loss: {(total_loss / total_samples):.4f} '
                                              f'| Batch accuracy: {(batch_correct / batch):.4f} '
                                              f'| Measured in intervals of {batch}')
                    pbar_eval.update(batch)
                    batch_correct = 0

        return correct / total_samples if total_samples > 0 else -1
