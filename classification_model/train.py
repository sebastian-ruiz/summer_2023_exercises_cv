# Imports
import torch
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from keypoint_classifier.model import KeyPointClassifier as KeyPointClassifierModel
from keypoint_classifier.dataloader import CustomDataset2D, CustomDataset3D, get_label_set, get_data_split


# Constants
CHECKER_SIZE = 0.035 #one mark is 3.5x3.5cm'set
TRAIN_PATH = '../results/recording4_hand_gestures_subset.json'
# TODO: exercise 3b -->
EPOCHS =
LR =
LR_STEP = [None]
NUM_FOLDS =
USE_2D =
BATCH_SIZE =
NUM_CLASSES=
# TODO: <-- exercise 3b




if USE_2D:
    INPUT_SIZE = 42
    CustomDataset = CustomDataset2D
    wfp = '2D'

else:
    INPUT_SIZE = 63
    CustomDataset = CustomDataset3D
    wfp = '3D'




# Get labels
labels_set = get_label_set(TRAIN_PATH)

fold_losses = []
# Get Train / Test Splits
for fold_id, (train_split, test_split) in enumerate(get_data_split(TRAIN_PATH, case=wfp, num_folds=NUM_FOLDS)):


    model = KeyPointClassifierModel(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)


    # Optimizer Set-Up
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = LR,
    )

    # Init Dataloader
    data_set = CustomDataset(train_data=train_split, labels_set=labels_set, transform=None, augment=True)
    train_loader = torch.utils.data.DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True)
   



    # Train Loop
    iteration_losses = []
    epoch_losses = []
    for epoch in tqdm(range(EPOCHS)):
        for datum in train_loader:
            inputs, labels = datum   

            outputs = model(inputs)

            optimizer.zero_grad()
            loss = model.loss_fn(outputs, labels)
            iteration_losses.append(loss.clone().detach())
            loss.backward()
            optimizer.step()
        
        if epoch in LR_STEP:
            lr = LR * (0.1 ** (LR_STEP.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
        epoch_losses.append(np.mean(iteration_losses))
        iteration_losses = []
        

    fold_losses.append(epoch_losses)


    # Save the Model
    torch.save(model.state_dict(), f'weights/classifier{wfp}_fold-{fold_id}_{EPOCHS}.pth')


    # Inference for ConfMat generation
    sm = nn.Softmax()
    def infer(points):
        points = torch.tensor(points).to(dtype=torch.float32)
        if USE_2D:
            points = (points / torch.tensor([512,384])) - 1
        else:
            std = torch.tensor([[0.1196, 0.0736, 0.0823]], dtype=torch.float32)
            mean = torch.tensor([[ 0.3239, -0.1417,  1.0394]], dtype=torch.float32)
            points = (points - mean) / std

        points = points.reshape(1, INPUT_SIZE)

        with torch.no_grad():
            output = sm(model(points)).squeeze()

        pred = torch.argmax(output)
        pred_label = labels_set[pred]

        return pred_label

    # Create ConfMat
    pred_ = []
    for i in range(len(test_split[0])):

        pred= infer(test_split[0][i])
        pred_.append(pred)
    gt_ = test_split[1]
    fig, ax = plt.subplots()
    print(gt_, pred_)
    cm = confusion_matrix(gt_, pred_)
    ax.set_xticklabels([''] + labels_set, )
    ax.set_yticklabels([''] + labels_set)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for Fold {fold_id}')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.show()


# Plot Loss Curve
for ID, fl in enumerate(fold_losses):
    plt.plot(fl, label=f'Fold ID {ID}')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, len(epoch_losses)+1, 500))
    plt.legend()
plt.show()
    
