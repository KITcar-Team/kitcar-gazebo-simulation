import argparse
from typing import Callable

from envyaml import EnvYAML
from tabulate import tabulate

from .drive_test_cmd import DriveTestCmd


def make_multiline(string: str, max_width: int):
    return "".join(
        [char + ("\n" if (i + 1) % max_width == 0 else "") for i, char in enumerate(string)]
    )


class Color:
    """Adds color to string for stdout.

    Attributes:
        success: Adds green to the text
        failure: Adds red to the text
    """

    def ansi(color_code: int) -> Callable[[str], str]:  # pylint: disable=no-self-argument
        """Returns a callable function to which a string can be parsed. The string will be
        colored in the color of the ansi color code.

        Args:
            color_code: The ansi color code in which the string should be colored. Take a
                        look at https://en.wikipedia.org/wiki/ANSI_escape_code#Colors

        Returns:
            A callable function colorize to which a string can be parsed
        """

        def colorize(s: str) -> str:
            """Colorizes data with the given color.

            Args:
                s: The string which should be colored

            Returns:
                A string in the given color
            """
            return f"\033[{color_code}m{s}\033[0m"

        return colorize

    success = ansi(92)
    failure = ansi(91)


class AutomatedDriveTest:
    """Automated Drive Testing."""

    def __init__(self, config: str):
        """Load config.

        Args:
            config: Path to config.
        """

        # Use envyaml so that users can set environment variables within the yaml file.
        data = EnvYAML(config)

        # Read parameters specified inside table_header
        self.table_data = [data["table_header"]]

        self.max_string_width = data["table_column_max_width"]

        # Get global tests parameters
        default_args = data["default_args"]

        # Loop over all tests in yaml
        # and parse paremeters to DriveTestCmd
        # Join args and default args.
        self.pipeline = [DriveTestCmd(**{**default_args, **args}) for args in data["tests"]]

    def execute(self):
        """Execute tests inside self.pipeline."""

        # Execute tests
        for job in self.pipeline:
            print(f"Starting {job.get_cmd()}")
            print("Waiting for the test to finish...")
            job.status, output = job.run()
            job.success = job.status == 0

            # Print log to console after job finishes
            # TODO: Print log while job is running
            print(output)

    def show_results(self):
        """Print the table of results to stdout."""

        # Colorize stdout
        SUCCESS = Color.success("Success")
        FAILED = Color.failure("Failure")

        # Fill table data
        for job in self.pipeline:
            row = []
            # Loop over first row in table (which should be the header)
            # and add desc or ros_args elements from DriveTestCmd object.
            # Also result gets added. (Which gets created during job execution)
            for header_elem in self.table_data[0]:
                if header_elem == "desc":
                    item = make_multiline(str(job.desc), self.max_string_width)
                elif header_elem == "result":
                    item = SUCCESS if job.success else FAILED
                elif header_elem == "conclusion":
                    must_succeed = SUCCESS if job.must_succeed else None
                    result = SUCCESS if job.success else FAILED
                    item = f"{result} (Expected: {must_succeed})"
                else:
                    item = make_multiline(
                        str(job.ros_args.get(header_elem)), self.max_string_width
                    )

                row.append(item)
            self.table_data.append(row)

        # Print table of results
        print("[RESULT]".ljust(80, "-"), end=2 * "\n")
        print(tabulate(self.table_data, headers="firstrow", tablefmt="fancy_grid"))

    def check(self):
        """Check if tests were successful (atleast the ones that should succeed)."""

        # Check if there is any failed job marked with must_succeed
        result_successful = not any(
            job.must_succeed and not job.success for job in self.pipeline
        )

        result_message = (
            Color.success("All roads marked with must_succeed have been run successfully.")
            if result_successful
            else Color.failure("Job has failed.")
        )

        print("\n", result_message, sep="")
        if not result_successful:
            exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run a number of drive test to automatically validate the car's"
            "behavior on predefined roads."
        )
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the config file.",
    )
    args = parser.parse_args()

    test = AutomatedDriveTest(args.config)
    test.execute()
    test.show_results()
    test.check()
