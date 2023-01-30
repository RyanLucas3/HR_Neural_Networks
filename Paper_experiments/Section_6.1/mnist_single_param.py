import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import math

import sys
sys.path.append("..")
sys.path.append("../..")

from pathlib import Path
import pandas as pd

from helper_functions import *
from HR import *
import time 
import random
import sys

torch.version.cuda == '11.3'

train_batch_size = 64
test_batch_size = 100

train_batches = math.floor(60000/64)

α_choice = 0.05
r_choice = 0.1
ϵ_choice = 0.2

model_name = f"alpha = {α_choice}, r = {r_choice}, eps = {ϵ_choice}"

noise = 0
gaussian_noise = 0
mis_specification = 0.25
frac = 0.05

normalisation_mean = [0.1307]
normalisation_std = [0.3081]
num_classes = 10

for iter in np.arange(0, 10):
    
    torch.manual_seed(iter)
    torch.cuda.manual_seed_all(iter)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # Volta: 100
    torch.backends.cudnn.enabled = True

    trainloader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('./Files', train=True, download=True, 
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     normalisation_mean, normalisation_std)
                                 ])),
      batch_size=train_batch_size, shuffle=False, num_workers=1, pin_memory = True, drop_last = True,)

    testloader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('./Files', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     normalisation_mean, normalisation_std)
                                 ])),
      batch_size=test_batch_size, shuffle=False, pin_memory = False)

    # Function to corrupt the data. That is, to sample less than the full data size and corrupt some of the labels
    # See the paper for more details.
    sampled_batches, data_points_to_corrupt, unique_labels = return_batches_to_corrupt(iter, 
                                                                                       train_batches, 
                                                                                       train_batch_size, 
                                                                                       frac, 
                                                                                       mis_specification, 
                                                                                       num_classes)


    device = 'cpu'

    # based on https://github.com/pytorch/examples/blob/main/mnist/main.py
    class Net(nn.Module):
        
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

    net = Net().to(device)

    criterion = nn.CrossEntropyLoss(reduction="none")

    optimizer = optim.Adam(net.parameters(), lr = 0.001)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    HR = HR_Neural_Networks(NN_model = net,
                            learning_approach = "HD",
                            train_batch_size = train_batch_size,
                            loss_fn = criterion,
                            normalisation_used = [normalisation_mean, normalisation_std],
                            α_choice = α_choice, 
                            r_choice = r_choice,
                            ϵ_choice = ϵ_choice,
                            adversarial_steps = 40,
                            adversarial_step_size = 0.01,
                            noise_set = 'l-2'
                            )

    ########## TRAINING ##################
    test_losses = []

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = []
        HR_losses = []
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):

            if batch_idx in sampled_batches:

                targets = corrupt_targets(batch_idx,
                targets,
                sampled_batches,
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


    adversarial_attack_test = torchattacks.PGDL2(net,
                                                 eps=noise,
                                                 alpha=0.01,
                                                 steps=40,
                                                 random_start=True,
                                                 eps_for_division=1e-10)

    adversarial_attack_test.set_normalization_used(
        mean=normalisation_mean, std=normalisation_std)

    def test(epoch):

        global best_acc
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

    nat_test_losses = []
    adv_test_losses = []
    gn_test_losses = []

    train_accuracies = []

    nat_test_accuracies = []
    adv_test_accuracies = []
    gn_test_accuracies = []

    start_epoch = 0
    min_epochs = max(220, int(220*(0.25/frac))) # If the data size is less than 0.25, we scale the number of epochs up (so the number of training steps remains constant)
    epochs = max(300, int(300*(0.25/frac))) # If the data size is less than 0.25, we scale the number of epochs up (so the number of training steps remains constant)
    end_epoch = start_epoch+epochs

    for epoch in range(start_epoch, end_epoch):

        stop = epoch == end_epoch-1

        train_loss, train_accuracy, HR_loss = train(epoch)

        if epoch % 20 == 0 or stop: # Collect metrics during training every 20 epochs or when the training is over.
            

            train_losses.append(train_loss)
            HR_losses.append(HR_loss)
            train_accuracies.append(train_accuracy)


            test_metrics = test(
                epoch)

            adv_test_losses.append(test_metrics["Adv Test Loss"])
            nat_test_losses.append(test_metrics["Nat Test Loss"])
            gn_test_losses.append(test_metrics["GN Test Loss"])
            adv_test_accuracies.append(test_metrics["Adv Test Accuracy"])
            nat_test_accuracies.append(test_metrics["Nat Test Accuracy"])
            gn_test_accuracies.append(test_metrics["GN Test Accuracy"])
            
            print(train_accuracy)
            print(test_metrics["GN Test Accuracy"])
            print(test_metrics["Adv Test Accuracy"])
            print(test_metrics["Nat Test Accuracy"])
            
        # Save checkpoint.
        best_train_loss = min(train_losses)

        if train_loss > best_train_loss:

            stopping_criterion += 1

        else:
            stopping_criterion = 0

        if stop:

            print('Saving..')

            models_path = f"mnist_single_parameter/{model_name}/frac_misspecified_{mis_specification}/frac_data_{frac}/iter_{iter}/"
            Path(models_path).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), models_path + "checkpoint.pt")

            losses = pd.DataFrame([train_losses, HR_losses, 
                                   nat_test_losses, adv_test_losses, gn_test_losses,
                                   train_accuracies, gn_test_accuracies,
                                   nat_test_accuracies, adv_test_accuracies, gn_test_accuracies]).T

            losses.columns = ["Training Loss", "Inflated Loss", 
                              "Natural Testing Loss", "Adversarial Testing Loss", "Gaussian Noise Testing Loss",
                              "Training Accuracy", 
                              "Natural Testing Accuracy", "Adversarial Testing Accuracy", "Gaussian Noise Testing Accuracy"]
            
            results_path = f"mnist_results/{model_name}/frac_misspecified_{mis_specification}/frac_data_{frac}/iter_{iter}/"
            Path(results_path).mkdir(parents=True, exist_ok=True)
            losses.to_csv(results_path + "metrics.csv")

            break


