import os
import pickle
import shutil
from itertools import chain

import torch
from hypothesis import given
from hypothesis.strategies import integers

from ..base_model import CycleGANNetworks
from ..generator import create_generator


@given(integers(1, 3), integers(1, 3))
def test_save(input_nc, output_nc):
    print("Test saving networks")

    net1 = create_generator(input_nc, output_nc, 64, netg="resnet_3blocks")
    net2 = pickle.loads(pickle.dumps(net1))

    nets = CycleGANNetworks(net1, net2)

    os.makedirs("temp/this_is_a_test", exist_ok=True)

    nets.save("temp/this_is_a_test/")

    assert os.path.isfile("temp/this_is_a_test/g_a_to_b.pth")
    assert os.path.isfile("temp/this_is_a_test/g_b_to_a.pth")

    shutil.rmtree("temp")


@given(integers(1, 3), integers(1, 3))
def test_load(input_nc, output_nc):
    print("Test loading networks")

    net1 = create_generator(input_nc, output_nc, 64, netg="resnet_3blocks")
    net2 = pickle.loads(pickle.dumps(net1))
    net3 = pickle.loads(pickle.dumps(net1))
    net4 = pickle.loads(pickle.dumps(net1))

    for param in chain(net1.parameters(), net2.parameters()):
        param.data = torch.rand(1).expand_as(param.data)

    nets1 = CycleGANNetworks(net1, net2)
    os.makedirs("temp/this_is_a_test", exist_ok=True)
    nets1.save("temp/this_is_a_test/")

    nets2 = CycleGANNetworks(net3, net4)

    def model_equals(model1, model2):
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if not torch.all(torch.eq(p1.data, p2.data)):
                return False
        return True

    assert not model_equals(nets1.g_a_to_b, nets2.g_a_to_b)
    assert not model_equals(nets1.g_b_to_a, nets2.g_b_to_a)

    nets2.load("temp/this_is_a_test/", torch.device("cpu"))

    assert model_equals(nets1.g_a_to_b, nets2.g_a_to_b)
    assert model_equals(nets1.g_b_to_a, nets2.g_b_to_a)

    shutil.rmtree("temp")


test_save()
test_load()
