import torch.nn as nn


# Model Definition
class KeyPointClassifier(nn.Module):
    def __init__(self, input_size=63, num_classes=18):
        super(KeyPointClassifier, self).__init__()
        self.layers = nn.Sequential(
            # TODO: exercise 3b -->

            # TODO: <-- exercise 3b

        )

        # Loss Term
        # TODO: exercise 3b -->
        self.loss_fn = None
        # TODO: <-- exercise 3b


    def forward(self, input):
        result = self.layers(input)
        return result
    