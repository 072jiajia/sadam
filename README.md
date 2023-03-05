# SAdam

SAdam (Scalable Adam) is an experimental optimizer that we
mathematically replace each parameters (p) in the model with
    ``` p = a * exp(b) ```
which means the convergence of Adam still holds since
we're implicitly optimizing the `a` and `b`

## Install
```
    pip install sadam
```

## Example
```python=
import torch
from torch.optim import Adam
from sadam import SAdam

optimizer_class = SAdam # Adam

inp = torch.tensor([[1., 0.], [0., 1.]])
tgt = torch.tensor([7000., 3000.])

model = torch.nn.Linear(2, 1, bias=False)
opt = optimizer_class(model.parameters(), lr=1e-3)

model.train()
for i in range(10000):
    pred = model(inp).view(-1)
    loss = torch.nn.functional.mse_loss(pred, tgt)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if i % 100 == 0:
        print(i, pred, loss)

```


## Contact Us
If you have any suggestions or questions about this work, please feel free to leave an issue or send an email to this [mailbox](mailto:jijiawu.cs@gmail.com).


## License

2023 Jijia Wu

This repository is licensed under the MIT license. See LICENSE for details.
