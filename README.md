# torch2moderngl

This is a small utility for transferring data between [ModernGL](https://github.com/moderngl/moderngl) 
and [PyTorch](https://pytorch.org)

# requirements
Install moderngl and pytorch (with cuda support)

# usage
```python
import moderngl
import torch
from torch2moderngl import texture2tensor, tensor2texture

ctx = moderngl.create_context(standalone=True)
texture = ctx.texture((200, 200), 4)  # must have 1, 2, or 4 components

tensor = texture2tensor(texture)  # create tensor from texture

tensor2texture(tensor, texture)  # transfer contents of tensor to existing texture

```
