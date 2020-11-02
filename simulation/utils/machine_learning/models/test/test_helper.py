import torch
from torch import nn

from .. import helper


def test_get_norm_layer():
    """Check if :py:func:`get_norm_layer` returns a valid layer."""
    features = 10
    batch_features = (1, 10, 10, 10)
    tensor = torch.rand(batch_features)

    def test_return_val(norm_type):
        norm_layer = helper.get_norm_layer(norm_type)(features)
        assert isinstance(norm_layer, nn.Module)

        result_tensor = norm_layer(tensor)
        assert result_tensor.shape == tensor.shape

    test_return_val("batch")
    test_return_val("instance")
    test_return_val("none")

    try:
        test_return_val("any_other_string")
        raise AssertionError("get_norm_layer should have failed due to wrong input.")
    except NotImplementedError:
        pass


def test_get_scheduler():
    """Check if :py:func:`get_scheduler` returns a scheduler."""
    module = nn.Linear(10, 10)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

    def test_return_val(lr_policy, **kwargs):
        scheduler = helper.get_scheduler(
            optimizer, lr_policy, lr_decay_iters=10, n_epochs=2, lr_step_factor=0.1
        )
        optimizer.step()
        scheduler.step(**kwargs)

    test_return_val(lr_policy="linear")
    test_return_val(lr_policy="step")
    test_return_val(lr_policy="plateau", metrics=1)
    test_return_val(lr_policy="cosine")

    try:
        test_return_val("any_other_string")
        raise AssertionError("get_scheduler should have failed due to wrong input.")
    except NotImplementedError:
        pass


def test_init_weights():
    """Check if :py:func:`init_weights` runs without errors."""
    module = nn.Linear(10, 10)

    helper.init_net(module, init_type="normal")
    helper.init_net(module, init_type="xavier")
    helper.init_net(module, init_type="kaiming")
    helper.init_net(module, init_type="orthogonal")

    try:
        helper.init_net(module, init_type="any_other_string")
        raise AssertionError("init_weights should have failed due to wrong input.")
    except NotImplementedError:
        pass


def test_set_requires_grad():
    """Check if :py:func:`set_requires_grad` correctly changes requires_grad."""
    module1 = nn.Linear(10, 10)
    module2 = nn.Linear(10, 10)

    helper.set_requires_grad(module1, requires_grad=False)
    assert not any(param.requires_grad for param in module1.parameters())

    helper.set_requires_grad(module1, requires_grad=True)
    assert all(param.requires_grad for param in module1.parameters())

    helper.set_requires_grad([module1, module2], requires_grad=False)
    assert not any(param.requires_grad for param in module1.parameters())
    assert not any(param.requires_grad for param in module2.parameters())

    helper.set_requires_grad([module1, module2], requires_grad=True)
    assert all(param.requires_grad for param in module1.parameters())
    assert all(param.requires_grad for param in module2.parameters())


def main():
    test_get_norm_layer()
    test_get_scheduler()
    test_init_weights()
    test_set_requires_grad()


if __name__ == "__main__":
    main()
