from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument(
            "--results_dir", type=str, default="./results/", help="saves results here."
        )
        parser.add_argument(
            "--aspect_ratio", type=float, default=1.0, help="aspect ratio of result images"
        )
        parser.add_argument(
            "--phase", type=str, default="test", help="train, val, test, etc"
        )
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default("crop_size"))
        self.isTrain = False
        return parser
