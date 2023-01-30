'''Train Robust NN on CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import random

import os
import argparse
import sys

from resnet import *
from helper_functions import *

import torchattacks
import pandas as pd
import time
import numpy as np
import cvxpy as cp

os.environ['MOSEKLM_LICENSE_FILE']="mosek.lic"

from HR import *

assert torch.cuda.is_available() == True
torch.version.cuda == '11.3'

normalisation_mean = [0.4914, 0.4822, 0.4465]
normalisation_std = [0.2023, 0.1994, 0.2010]

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(normalisation_mean, normalisation_std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(normalisation_mean, normalisation_std),
])

train_batch_size = 128
train_batches = 390
test_batch_size = 100

trainset = torchvision.datasets.CIFAR10(
    root='./Files', train=True, download=False, transform=transform_train)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=train_batch_size, shuffle=False, num_workers=20)

testset = torchvision.datasets.CIFAR10(
    root='./Files', train=False, download=False, transform=transform_test)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=test_batch_size, shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

num_classes = len(classes)

α_choice = 0.05
r_choice = 0.0
ϵ_choice = 0.0

# Robustness specifications
model_name = f"alpha = {α_choice}, r = {r_choice}, eps = {ϵ_choice}"

for iter in np.arange(0, 10):
    
    torch.manual_seed(iter)
    torch.cuda.manual_seed_all(iter)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # Volta: 100
    torch.backends.cudnn.enabled = True

    # Error sources in the data
    frac = 0.25 # What fraction of the training data should we consider as a finite random sample?
    mis_specification = 0.1 # What fraction of the training labels are misspecified?
    noise = 0.1  # How much adversarial noise is in the data?
    gaussian_noise = 0.5  # How much gaussian noise is in the data?

    # Data & Training Specifications
    epochs = 300
    lr = 0.01
    resume = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start_epoch = 0 # start from epoch 0 or last checkpoint epoch

    # Implementing the error sources.
    # Statistical error - sample less than the full dataset
    
    # Function to corrupt the data. That is, to sample less than the full data size and corrupt some of the labels
    # See the paper for more details.
    sampled_batches, data_points_to_corrupt, unique_labels = return_batches_to_corrupt(iter, 
                                                                                       train_batches, 
                                                                                       train_batch_size, 
                                                                                       frac, 
                                                                                       mis_specification, 
                                                                                       num_classes)
    
    np.random.seed(iter)
    training_batches = np.random.choice(sampled_batches, size = int(0.7*len(sampled_batches)), replace = False)
    validation_batches = [i for i in sampled_batches if i not in training_batches]
    
    
    # Final Fitting
    net = ResNet18(iter)
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    ########### TRAINING ##################

    criterion = nn.CrossEntropyLoss(reduction="none")

    optimizer = optim.Adam(net.parameters(), lr=lr)
       
    HR = HR_Neural_Networks(NN_model = net,
                        learning_approach = "HD",
                        train_batch_size = train_batch_size,
                        loss_fn = criterion,
                        normalisation_used = [normalisation_mean, normalisation_std],
                        α_choice = α_choice, 
                        r_choice = r_choice,
                        ϵ_choice = ϵ_choice
                        )

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = []
        HR_losses = []
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):

            if batch_idx in training_batches:
 
                targets = corrupt_targets(batch_idx,
                targets,
                training_batches,
                train_batch_size,
                train_batches,
                data_points_to_corrupt,
                unique_labels)

                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
            
                if (α_choice != 0 or r_choice != 0 or ϵ_choice != 0):
                    HR_loss = HR.HR_criterion(inputs, targets, device) # Resolving to find the new weighted loss function
                else:
                    outputs = net(inputs)
                    HR_loss = torch.sum(criterion(outputs, targets))/train_batch_size
                    
                HR_loss.backward()

                outputs = net(inputs)

                optimizer.step()
                
                HR_losses.append(HR_loss.cpu().detach().numpy())

                loss = torch.sum(criterion(outputs, targets))/train_batch_size
                train_loss.append(loss.item())
                
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        training_accuracy = correct/total

        return np.mean(train_loss), training_accuracy, np.mean(HR_losses)


    ########### TESTING ##################

    adversarial_attack_test = torchattacks.PGDL2(net,
                                                 eps=noise,
                                                 alpha=0.2,
                                                 steps=10,
                                                 random_start=True,
                                                 eps_for_division=1e-10)

    adversarial_attack_test.set_normalization_used(
        mean=normalisation_mean, std=normalisation_std)
    
    def validate(epoch):

        net.eval()

        adv_test_loss = []
        nat_test_loss = []
        gn_test_loss = []

        adv_correct = 0
        nat_correct = 0
        gn_correct = 0

        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            
            if batch_idx in validation_batches:
                
                targets = corrupt_targets(batch_idx,
                targets,
                validation_batches,
                train_batch_size,
                train_batches, 
                data_points_to_corrupt,
                unique_labels)

                inputs = inputs.to(device)
                targets = targets.to(device)

                if noise > 0:
                    adv = adversarial_attack_test(inputs, targets)

                else:
                    adv = inputs

                if gaussian_noise > 0:
                    gn = gaussian_noise * torch.randn(*inputs.shape)
                    gn = gn.to(device)
                    gn_inputs = inputs + gn

                with torch.no_grad():

                    if gaussian_noise > 0:

                        gn_outputs = net(gn_inputs)
                        gn_loss_vec = criterion(gn_outputs, targets)
                        gn_loss = torch.sum(gn_loss_vec)
                        gn_test_loss.append(gn_loss.item()/test_batch_size)
                        gn_predictions = gn_outputs.max(1)[1]
                        gn_correct += gn_predictions.eq(targets).sum().item()


                    if noise > 0:

                        adv_outputs = net(adv)
                        adv_loss_vec = criterion(adv_outputs, targets)
                        adv_loss = torch.sum(adv_loss_vec)
                        adv_test_loss.append(adv_loss.item()/test_batch_size)
                        adv_predictions = adv_outputs.max(1)[1]
                        adv_correct += adv_predictions.eq(targets).sum().item()
                        adv_outputs = net(adv)


                    nat_outputs = net(inputs)
                    nat_loss_vec = criterion(nat_outputs, targets)
                    nat_loss = torch.sum(nat_loss_vec)
                    nat_test_loss.append(nat_loss.item()/test_batch_size)
                    nat_predictions = nat_outputs.max(1)[1]
                    nat_correct += nat_predictions.eq(targets).sum().item()

                    total += targets.size(0)

        outputs = {"Adv Val Loss": -1, "Adv Val Accuracy": -1,
                   "GN Val Loss": -1, "GN Val Accuracy": -1,
                   "Nat Val Loss": -1, "Nat Val Accuracy": -1}

        if noise > 0:

            outputs["Adv Val Loss"] = np.mean(adv_test_loss)

            adv_accuracy = adv_correct/total

            outputs["Adv Val Accuracy"] = adv_accuracy


        if gaussian_noise > 0:

            outputs["GN Val Loss"] = np.mean(gn_test_loss)

            gn_accuracy = gn_correct/total

            outputs["GN Val Accuracy"] = gn_accuracy

        nat_accuracy = nat_correct/total

        outputs["Nat Val Loss"] = np.mean(nat_test_loss)
        outputs["Nat Val Accuracy"] = nat_accuracy

        return outputs

    def test(epoch):

        net.eval()

        adv_test_loss = []
        nat_test_loss = []
        gn_test_loss = []

        adv_correct = 0
        nat_correct = 0
        gn_correct = 0

        total = 0

        for batch_idx, (inputs, targets) in enumerate(testloader):

            inputs = inputs.to(device)
            targets = targets.to(device)

            if noise > 0:
                adv = adversarial_attack_test(inputs, targets)

            else:
                adv = inputs

            if gaussian_noise > 0:
                gn = gaussian_noise * torch.randn(*inputs.shape)
                gn = gn.to(device)
                gn_inputs = inputs + gn

            with torch.no_grad():

                if gaussian_noise > 0:

                    gn_outputs = net(gn_inputs)
                    gn_loss_vec = criterion(gn_outputs, targets)
                    gn_loss = torch.sum(gn_loss_vec)
                    gn_test_loss.append(gn_loss.item()/test_batch_size)
                    gn_predictions = gn_outputs.max(1)[1]
                    gn_correct += gn_predictions.eq(targets).sum().item()


                if noise > 0:

                    adv_outputs = net(adv)
                    adv_loss_vec = criterion(adv_outputs, targets)
                    adv_loss = torch.sum(adv_loss_vec)
                    adv_test_loss.append(adv_loss.item()/test_batch_size)
                    adv_predictions = adv_outputs.max(1)[1]
                    adv_correct += adv_predictions.eq(targets).sum().item()
                    adv_outputs = net(adv)


                nat_outputs = net(inputs)
                nat_loss_vec = criterion(nat_outputs, targets)
                nat_loss = torch.sum(nat_loss_vec)
                nat_test_loss.append(nat_loss.item()/test_batch_size)
                nat_predictions = nat_outputs.max(1)[1]
                nat_correct += nat_predictions.eq(targets).sum().item()

                total += targets.size(0)

        outputs = {"Adv Test Loss": -1, "Adv Test Accuracy": -1,
                   "GN Test Loss": -1, "GN Test Accuracy": -1,
                   "Nat Test Loss": -1, "Nat Test Accuracy": -1}

        if noise > 0:

            outputs["Adv Test Loss"] = np.mean(adv_test_loss)

            adv_accuracy = adv_correct/total

            outputs["Adv Test Accuracy"] = adv_accuracy


        if gaussian_noise > 0:

            outputs["GN Test Loss"] = np.mean(gn_test_loss)

            gn_accuracy = gn_correct/total

            outputs["GN Test Accuracy"] = gn_accuracy

        nat_accuracy = nat_correct/total

        outputs["Nat Test Loss"] = np.mean(nat_test_loss)
        outputs["Nat Test Accuracy"] = nat_accuracy

        return outputs

    # RUNNING TRAINING AND TESTING

    train_losses = []
    HR_losses = []
    
    nat_val_losses = []
    adv_val_losses = []
    gn_val_losses = []
    
    nat_test_losses = []
    adv_test_losses = []
    gn_test_losses = []

    train_accuracies = []
                                        
    nat_val_accuracies = []
    adv_val_accuracies = []
    gn_val_accuracies = []
    
    nat_test_accuracies = []
    adv_test_accuracies = []
    gn_test_accuracies = []

    stopping_criterion = 0
    timing = []

    min_epochs = 220
    end_epoch = start_epoch+epochs

    for epoch in range(start_epoch, end_epoch):

        stop = (stopping_criterion >= 6 and epoch >= min_epochs) or epoch == end_epoch-1

        train_loss, train_accuracy, HR_loss = train(epoch)

        if epoch % 20 == 0 or stop: # Collect metrics during training every 20 epochs or when the training is over.
            
            print("Train accuracy")
            print(train_accuracy)
            print("--------------")
            
            print("Train Loss")
            print(train_loss)
            print("--------------")
            
            train_losses.append(train_loss)
            HR_losses.append(HR_loss)
            train_accuracies.append(train_accuracy)
            
            val_metrics = validate(
                epoch)
            
            print("Validation Accuracy")
            print(val_metrics["Nat Val Accuracy"])
            print("--------------")
            
            print("Validation Loss")
            print(val_metrics['Nat Val Loss'])
            print("--------------")
                  
            adv_val_losses.append(val_metrics["Adv Val Loss"])
            nat_val_losses.append(val_metrics["Nat Val Loss"])
            gn_val_losses.append(val_metrics["GN Val Loss"])
            adv_val_accuracies.append(val_metrics["Adv Val Accuracy"])
            nat_val_accuracies.append(val_metrics["Nat Val Accuracy"])
            gn_val_accuracies.append(val_metrics["GN Val Accuracy"])
                                        
            test_metrics = test(
                epoch)
                 
            print("Testing Accuracy")
            print(test_metrics["Nat Test Accuracy"])
            print("--------------")
            
            print("Testing Loss")
            print(test_metrics['Nat Test Loss'])
            print("--------------")

                                        
            adv_test_losses.append(test_metrics["Adv Test Loss"])
            nat_test_losses.append(test_metrics["Nat Test Loss"])
            gn_test_losses.append(test_metrics["GN Test Loss"])
            adv_test_accuracies.append(test_metrics["Adv Test Accuracy"])
            nat_test_accuracies.append(test_metrics["Nat Test Accuracy"])
            gn_test_accuracies.append(test_metrics["GN Test Accuracy"])
        
        # Save checkpoint.
        best_train_loss = min(train_losses)

        if train_loss > best_train_loss:
            
            stopping_criterion += 1

        else:
            stopping_criterion = 0

        if stop:

            print('Saving..')
            
            models_path = f"HD_models_only_stat_error/{model_name}/frac_misspecified_{mis_specification}/frac_data_{frac}/iter_{iter}/"
            Path(models_path).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), models_path + "checkpoint.pt")

            losses = pd.DataFrame([train_losses, HR_losses, 
                                   nat_val_losses, adv_val_losses, gn_val_losses,
                                   nat_test_losses, adv_test_losses, gn_test_losses,
                                   train_accuracies, 
                                   nat_val_accuracies, adv_val_accuracies, gn_test_accuracies,
                                   nat_test_accuracies, adv_test_accuracies, gn_test_accuracies]).T
                                        
            losses.columns = ["Training Loss", "Inflated Loss", 
                              "Natural Validation Loss", "Adversarial Validation Loss", "Gaussian Noise Validation Loss",
                              "Natural Testing Loss", "Adversarial Testing Loss", "Gaussian Noise Testing Loss",
                              "Training Accuracy", 
                              "Natural Validation Accuracy", "Adversarial Testing Accuracy", "Gaussian Noise Testing Accuracy",
                              "Natural Testing Accuracy", "Adversarial Testing Accuracy", "Gaussian Noise Testing Accuracy"]
            
            results_path = f"HD_results_only_stat_error/{model_name}/frac_misspecified_{mis_specification}/frac_data_{frac}/iter_{iter}/"
            Path(results_path).mkdir(parents=True, exist_ok=True)
            losses.to_csv(results_path + "metrics.csv")
            
            break
