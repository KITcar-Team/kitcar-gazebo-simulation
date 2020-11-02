# Run all tests in this package.
from . import (  # noqa: F401
    test_helper,
    test_resnet_block,
    test_resnet_generator,
    test_wgan_critic,
)

test_helper.main()
test_resnet_block.main()
test_resnet_generator.main()
test_wgan_critic.main()
