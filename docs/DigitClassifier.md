# Let's build a digit classifier!

# Load MNIST training and test set


```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.MNIST(
    'data/',
    train = True,
    transform = ToTensor(),
    download = True)

test_data = datasets.MNIST(
    'data/',
    train = False,
    transform = ToTensor(),
    download = True)
    
print(len(training_data), len(test_data))
```

# Look at the data


```python
(image, label) = training_data[0]
print(image)
print(image.shape)
print(label)
```

# Visualize using matplotlib


```python
import random
from matplotlib import pyplot

(image, label) = random.choice(training_data)
pyplot.imshow(image[0])
print(label)
```

# DataLoader


```python

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)

for (index, (features, labels)) in enumerate(train_dataloader):
    if index == 10: break
    print(features.shape, features.dtype)
    print(labels)
```

# Feed-Forward Neural Network


```python
from torch import nn

class DigitClassifierFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_hidden = 8
        # layers with weights
        self.layers = nn.Sequential(
            nn.Flatten(1),
            # 28 * 28 * 8 = 6272
            nn.Linear(28 * 28, self.num_hidden),
            nn.ReLU(),
            nn.Linear(self.num_hidden, 10),
            nn.LogSoftmax(dim=1))

    # input: [N, 1, 28, 28]
    def forward(self, input):
        log_prob = self.layers(input)
        return log_prob
```

## Instantiate the model


```python
digit_classifier_ff = DigitClassifierFF()
```

## Try it out


```python
import torch
import random
from matplotlib import pyplot

def visualize(model, data):
    (image, correct_label) = random.choice(test_data)
    # image: 1 x 28 x 28
    print(f'Correct answer: {correct_label}')
    log_prob = model(image.reshape(1, 1, 28, 28))
    # log_prob: 1 x 10
    prob = log_prob[0].exp().tolist()
    fig, axes = pyplot.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image[0])
    axes[1].bar(list(range(10)), prob)
    pyplot.show()
```


```python
visualize(digit_classifier_ff, test_data)
```

## Evaluate the model on test data


```python
from torch.utils.data import DataLoader

def evaluate(model, data):
    cost_fn = nn.NLLLoss()
    data_loader = DataLoader(data, batch_size = 32)
    cost = 0.0
    correct = 0
    with torch.no_grad():
        for (images, correct_labels) in data_loader:
            log_prob = model(images)
            cost += len(images) * cost_fn(log_prob, correct_labels).item()
            correct += (log_prob.argmax(dim=1) == correct_labels).sum().item()
    cost /= len(data)
    correct /= len(data)
    print(f'Evaluation cost: {cost:.8f} correct: {100 * correct:.2f}%')
    return cost
```


```python
evaluate(digit_classifier_ff, test_data)
```

### Base cost for random guessing


```python
torch.tensor(10.0).log()
```

## Train DigitClassfierFF


```python
from torch import optim

def training_epoch(model, data):
    cost_fn = nn.NLLLoss()
    data_loader = DataLoader(data, batch_size=32, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr = 0.001)
    total_cost = 0.0
    for (images, labels) in data_loader:
        log_prob = model(images)
        cost = cost_fn(log_prob, labels)
        total_cost += len(images) * cost.item()
        # backpropagation
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
    total_cost /= len(data)
    print(f'Training cost: {total_cost:.8f}')
    return total_cost
```


```python
def train(model):
    training_costs = []
    validation_costs = []
    for epoch in range(10):
        training_cost = training_epoch(model, training_data)
        validation_cost = evaluate(model, test_data)
        training_costs.append(training_cost)
        validation_costs.append(validation_cost)
    return training_costs, validation_costs
```


```python
digit_classifier_ff = DigitClassifierFF()
training_costs, validation_costs = train(digit_classifier_ff)
```

## Visualize training progress


```python
def visualize_cost(training_costs, validation_costs):
    pyplot.plot(range(len(training_costs)), training_costs, label = 'training')
    pyplot.plot(range(len(validation_costs)), validation_costs, label = 'validation')
    pyplot.legend()
    pyplot.show()
```


```python
visualize_cost(training_costs, validation_costs)
```

## Try it out again


```python
visualize(digit_classifier_ff, test_data)
```

## Inspect first layer weights


```python
print(digit_classifier_ff.layers[1])
```


```python
print(digit_classifier_ff.layers[1].weight.shape)
```


```python
index = 7
image = digit_classifier_ff.layers[1].weight[index].detach().reshape(28, 28)
print(image.min(), image.max())
pyplot.imshow(image)
```

---

# Convolutional Neural Network


```python
class DigitClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels_1 = 8
        self.channels_2 = 8
        self.channels_3 = 8
        self.num_hidden = 16
        self.layers = nn.Sequential(
            # 5^2 * 1 * 8 = 200 params
            nn.Conv2d(1, self.channels_1, 5, padding='same'),
            # N x 14 x 14 x 8
            nn.MaxPool2d(2),
            nn.ReLU(),
            # 5^2 * 8 * 8 = 1600 params
            nn.Conv2d(self.channels_1, self.channels_2, 5, padding='same'),
            # N x 7 x 7 x 8
            nn.MaxPool2d(2),
            nn.ReLU(),
            # 5^2 * 8 * 8 = 1600 params
            nn.Conv2d(self.channels_2, self.channels_3, 5, padding='same'),
            # N x 4 x 4 x 8
            nn.MaxPool2d(2, ceil_mode=True),
            nn.ReLU(),
            # N x 128
            nn.Flatten(1),
            # 128 * 16 = 1024 params
            nn.Linear(4 * 4 * self.channels_3, self.num_hidden),
            nn.ReLU(),
            # 16 * 10 = 160 params
            nn.Linear(self.num_hidden, 10),
            nn.LogSoftmax(dim=1))

    # input: [N, 1, 28, 28]
    def forward(self, input):
        log_prob = self.layers(input)
        return log_prob
```

## Try out untrained CNN


```python
digit_classifier_cnn = DigitClassifierCNN()
```


```python
visualize(digit_classifier_cnn, test_data)
```


```python
evaluate(digit_classifier_cnn, test_data)
```

## Train CNN


```python
digit_classifier_cnn = DigitClassifierCNN()
training_costs, validation_costs = train(digit_classifier_cnn)
```


```python
visualize_cost(training_costs, validation_costs)
```

## Try out trained CNN


```python
visualize(digit_classifier_cnn, test_data)
```
