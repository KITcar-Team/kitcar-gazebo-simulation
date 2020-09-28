"""Perform some basic tests for the WGAN critic."""

import itertools
import pickle

import torch
from hypothesis import given, settings
from hypothesis.strategies import floats, integers

from ..wasserstein_critic import WassersteinCritic


@settings(deadline=None)
@given(floats(min_value=-0.1, max_value=0), floats(min_value=0, max_value=0.1))
def test_weight_clipping(lower, upper):
    if lower == upper:
        pass
    bounds = (lower, upper)
    print(f"Test weight clipping with bounds: {bounds}")
    # Since weights are initialized randomly, some are out of
    # bounds for almost certain...
    critic = WassersteinCritic(input_nc=1, height=32, width=32)

    critic._clip_weights(bounds)

    # Loop through parameters and ensure that clipping works
    for t in critic.parameters():
        lower = bounds[0] * torch.ones_like(t)
        upper = bounds[1] * torch.ones_like(t)
        assert torch.all(t >= lower).item()
        assert torch.all(t <= upper).item()


@settings(deadline=None)
@given(
    integers(min_value=1, max_value=16),
    integers(min_value=1, max_value=3),
    integers(min_value=3, max_value=6),
    integers(min_value=3, max_value=6),
)
def test_forward(batch_size, input_nc, height_log_2, width_log_2):
    height = 2 ** height_log_2
    width = 2 ** width_log_2
    print(
        f"Test critic input with batch_size:{batch_size},"
        f"input_nc:{input_nc}, height:{height}, width:{width}"
    )
    critic = WassersteinCritic(input_nc=input_nc, height=height, width=width)

    input = torch.rand(batch_size, input_nc, height, width)
    output = critic(input)
    assert output.shape == torch.Size([batch_size, 1])


@settings(deadline=None)
def test_optimization_step():
    """Testing very basic functionality of optimizing.

    Testing if the optimization works is hard. Here, some very basic things are tested:

    * Does the wasserstein distance increase when running the optimization?
    * Is the distance close to zero, if random distributions are given
        and the generator is the identity?
    * Is the generator unchanged?
    """
    # 1. Distances close to zero if both batches are randomly sampled each iteration
    #    and generator is the identity
    batch_size, input_nc, height, width = 16, 2, 32, 32

    critic = WassersteinCritic(input_nc=input_nc, height=height, width=width)
    optimizer = torch.optim.RMSprop(critic.parameters(), lr=0.00005)
    generator = torch.nn.Identity()

    iterations = 100

    distances = [
        critic.perform_optimization_step(
            generator,
            optimizer,
            torch.rand(batch_size, input_nc, height, width),
            torch.rand(batch_size, input_nc, height, width),
        )
        for _ in range(iterations)
    ]

    assert sum(distances) / iterations < 0.1

    # 2. Distances increase if batches are constant and critic should learn which one
    # is which and generator is the identity
    critic = WassersteinCritic(input_nc=input_nc, height=height, width=width)
    optimizer = torch.optim.RMSprop(critic.parameters(), lr=0.00005)
    generator = torch.nn.Identity()

    batch_critic, batch_generator = (
        torch.rand(batch_size, input_nc, height, width),
        torch.rand(batch_size, input_nc, height, width),
    )

    distances = [
        critic.perform_optimization_step(
            generator, optimizer, batch_critic, batch_generator
        )
        for _ in range(iterations)
    ]

    assert sum(distances[iterations // 2 :]) > sum(distances[: iterations // 2])

    # 3. Test if the generator's parameters are modified by the critics optimization
    #    Should not happen!
    critic = WassersteinCritic(input_nc=input_nc, height=height, width=width)

    generator = torch.nn.Conv2d(input_nc, input_nc, kernel_size=3)
    generator_clone = pickle.loads(pickle.dumps(generator))
    in_params = list(generator_clone.parameters())

    # Add generator parameters to optimizer as well to create a scenario where the critic's
    # optimization could change the generator's parameters. It needs to not do that.
    optimizer = torch.optim.RMSprop(
        itertools.chain(critic.parameters(), generator.parameters()), lr=0.00005
    )

    batch_critic, batch_generator = (
        torch.rand(batch_size, input_nc, height, width),
        torch.rand(batch_size, input_nc, height, width),
    )

    for _ in range(10):
        critic.perform_optimization_step(
            generator, optimizer, batch_critic, batch_generator
        )

    out_params = list(generator.parameters())

    assert len(in_params) == len(out_params)  # Shouldn't fail anyways...

    for i, o in zip(in_params, out_params):
        assert torch.all(i == o)


if __name__ == "__main__":
    test_weight_clipping()
    test_forward()
    test_optimization_step()
