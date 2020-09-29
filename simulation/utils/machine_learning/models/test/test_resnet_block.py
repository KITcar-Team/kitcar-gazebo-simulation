import torch

from .. import helper, resnet_block


def test_creating_resnet_block(norm_type, **kwargs):

    kwargs["norm_layer"] = helper.get_norm_layer(norm_type)
    tensor = torch.rand((3, kwargs["dim"], 30, 30))
    block = resnet_block.ResnetBlock(**kwargs)

    assert isinstance(block(tensor), torch.Tensor)


test_creating_resnet_block(
    dim=4,
    padding_type="replicate",
    norm_type="instance",
    use_dropout=True,
    use_bias=True,
)

test_creating_resnet_block(
    dim=2,
    padding_type="reflect",
    norm_type="none",
    use_dropout=False,
    use_bias=False,
    n_conv_layers=3,
    dilations=[2, 4, 2],
)
test_creating_resnet_block(
    dim=4,
    padding_type="replicate",
    norm_type="instance",
    use_dropout=False,
    use_bias=False,
    n_conv_layers=6,
    dilations=None,
)
