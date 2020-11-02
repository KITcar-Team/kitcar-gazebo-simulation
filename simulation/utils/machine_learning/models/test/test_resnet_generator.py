import torch

from .. import helper, resnet_generator


def test_resnet_generator(norm_type, **kwargs):
    kwargs["norm_layer"] = helper.get_norm_layer(norm_type)

    gen = resnet_generator.ResnetGenerator(**kwargs)

    batch_size = 4
    w = pow(2, 5)
    h = pow(2, 7)
    test_input = torch.rand(batch_size, kwargs["input_nc"], w, h)

    assert gen(test_input).shape == (batch_size, kwargs["output_nc"], w, h), (
        f"Shapes differ: {gen(test_input).shape}"
        f'vs expected {(batch_size, kwargs["output_nc"], w, h)}'
    )


def main():
    test_resnet_generator(
        input_nc=2,
        output_nc=2,
        padding_type="reflect",
        norm_type="none",
        use_dropout=False,
        activation=torch.nn.ReLU(),
        conv_layers_in_block=3,
        dilations=[2, 4, 2],
    )
    test_resnet_generator(
        input_nc=2,
        output_nc=3,
        padding_type="reflect",
        norm_type="instance",
        use_dropout=True,
        activation=torch.nn.Tanh(),
        conv_layers_in_block=3,
        dilations=None,
    )


if __name__ == "__main__":
    main()
