"""Definition of the data tests.

Images required to run the tests are small. They are therefore equipped with an extension
that is not stored in git lfs but directly in git.

Whenever this module is executed, all of the tests included below are run.
"""

from . import test_init  # noqa: 402
from . import test_labeled_data  # noqa: 402

test_labeled_data.main()
test_init.main()
