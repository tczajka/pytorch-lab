# Intro to PyTorch Tensors


```python
import torch
```

## Creating vectors


```python
a = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
print(a)
print(a.shape)
```


```python
print(torch.zeros(10))
print(torch.ones(10))
```


```python
torch.rand(10000)
```

## Vector operations


```python
a.sum()
```


```python
print(a.sum().shape)
print(a.sum().item())
```


```python
b = torch.tensor([10, 11, 12, 13], dtype=torch.float32)
print(b)
```


```python
a + b
```


```python
a * b
```


```python
a ** 2
```


```python
a.sqrt()
```


```python
a.log()
```


```python
torch.cat([a, b])
```


```python
a.dot(b)
```


```python
a + 17
```


```python
a.argmax(0)
```

## Matrices


```python
A = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print(A)
print(A.shape)
```


```python
A[0]
```


```python
A[:, 1]
```


```python
A[0, [0, 2]]
```


```python
B = torch.rand([3, 4])
print(B)
```


```python
B @ a  # matrix-vector multiplication
```


```python
A @ B  # matrix-matrix multiplication
```


```python
B.sum(dim=1)
```


```python
B.mean(dim=0)
```


```python
A.reshape(3, 2)
```


```python
A.flatten()
```

## 3D tensors


```python
X = torch.tensor([
    [[1,2], [3,4]],
    [[5,6], [7,8]],
    [[9,10], [11,12]]], dtype=torch.float32)
print(X)
```


```python
X @ A
```

# Automatic differentiation


```python
a = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
b = torch.tensor([5.0, 6.0, 7.0, 8.0], requires_grad=True)
c = a * b
print(c)
```


```python
d = c ** 2
print(d)
```


```python
e = d.sum()
print(e)
```


```python
e.backward()
```


```python
print(a.grad)
print(b.grad)
```


```python
from torch import nn
layer = nn.Softmax(dim=0)
```


```python
a = torch.tensor([-5.0, 1.0, 0.3])
layer(a)
```
