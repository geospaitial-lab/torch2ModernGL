import torch
from . import _cuda_functions


_dtype_int = {"f1": 0, "u1": 0, "f4": 1, "i1": 2, "i2": 3, "i4": 4}
_tex2torch_dtype = {"f1": torch.uint8, "f4": torch.float32, "i1": torch.int8, "i2": torch.int16, "i4": torch.int32}
_dtype_bytes = {"f1": 1, "f4": 4, "i1": 1, "i2": 2, "i4": 4}


def texture2tensor(texture):
    """
    Converts a moderngl texture object to a pytorch tensor using cuda-opengl interop.
    :param moderngl.Texture texture: A moderngl.Texture object to be converted. Must be GL_TEXTURE_2D.
    :rtype: torch.Tensor
    :return: The torch.Tensor Object created.
    """

    gl_object = texture.glo
    width, height = texture.size
    components = texture.components
    if components == 3:
        components = 4
    dtype = texture.dtype
    if dtype not in _dtype_int.keys():
        raise ValueError("Unsupported texture format! Must be in ['f1', 'f4', 'i1', 'i2', 'i4']!")
    int_dtype = _dtype_int[dtype]

    tensor = _cuda_functions.gl_texture_to_torch(gl_object, width, height, components, int_dtype)

    return torch.flip(tensor, dims=[0])


def tensor2texture(tensor, texture):
    """
    Copies the content of a pytorch tensor to a moderngl texture object using cuda-opengl interop.
    :param torch.Tensor tensor: The torch.Tensor Object to copy from.
    :param moderngl.Texture texture: A moderngl.Texture object to copy to. Must be GL_TEXTURE_2D. Must have 1, 2 or 4
           components.
    """

    gl_object = texture.glo
    width, height = texture.size
    components = texture.components
    element_bytes = components * _dtype_bytes[texture.dtype]
    if components == 3:
        raise ValueError("3 component textures are not allowed!")

    tensor = tensor[..., :components]

    tensor_components = tensor.size(dim=-1)

    if tensor_components < components:
        ones = torch.ones(*tensor.size()[:-1], components - tensor_components, device=tensor.device)
        tensor = torch.concat([tensor, ones], dim=-1)

    tensor = tensor.to(_tex2torch_dtype[texture.dtype])

    tensor = torch.flip(tensor, dims=[0])

    _cuda_functions.torch_to_gl_texture(tensor, gl_object, width, height, element_bytes)
