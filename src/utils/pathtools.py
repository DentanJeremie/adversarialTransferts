import collections
import datetime
import os
from pathlib import Path
import typing as t


class CustomizedPath():

    def __init__(self):
        self._root = Path(__file__).parent.parent.parent

        # Logs initialized
        self._initialized_loggers = collections.defaultdict(bool)

        # Datasets promises
        self._train = None
        self._test = None
        self._sample = None

# ------------------ UTILS ------------------

    def remove_prefix(input_string: str, prefix: str) -> str:
        """Removes the prefix if exists at the beginning in the input string
        Needed for Python<3.9
        
        :param input_string: The input string
        :param prefix: The prefix
        :returns: The string without the prefix
        """
        if prefix and input_string.startswith(prefix):
            return input_string[len(prefix):]
        return input_string

    def as_relative(self, path: t.Union[str, Path]) -> Path:
        """Removes the prefix `self.root` from an absolute path.

        :param path: The absolute path
        :returns: A relative path starting at `self.root`
        """
        if type(path) == str:
            path = Path(path)
        return Path(CustomizedPath.remove_prefix(path.as_posix(), self.root.as_posix()))

    def mkdir_if_not_exists(self, path: Path, gitignore: bool=False) -> Path:
        """Makes the directory if it does not exists

        :param path: The input path
        :param gitignore: A boolean indicating if a gitignore must be included for the content of the directory
        :returns: The same path
        """
        path.mkdir(parents=True, exist_ok = True)

        if gitignore:
            with (path / '.gitignore').open('w') as f:
                f.write('*\n!.gitignore')

        return path

# ------------------ MAIN FOLDERS ------------------

    @property
    def root(self):
        return self._root

    @property
    def data(self):
        return self.mkdir_if_not_exists(self.root / 'data', gitignore=True)

    @property
    def output(self):
        return self.mkdir_if_not_exists(self.root / 'output', gitignore=True)

    @property
    def logs(self):
        return self.mkdir_if_not_exists(self.root / 'logs', gitignore=True)

# ------------------ LOGS ------------------

    def get_log_file(self, logger_name: str) -> Path:
        """Creates and initializes a logger.

        :param logger_name: The logger name to create
        :returns: A path to the `logger_name.log` created and/or initialized file
        """
        file_name = logger_name + '.log'
        result = self.logs / file_name

        # Checking if exists
        if not os.path.isfile(result):
            with result.open('w') as f:
                pass

        # Header for new log
        if not self._initialized_loggers[logger_name]:
            with result.open('a') as f:
                f.write(f'\nNEW LOG AT {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n')
            self._initialized_loggers[logger_name] = True

        return result

# ------------------ TINY IMAGENET ------------------
    
    @property
    def tiny_imagenet(self):
        return self.data / 'tiny-imagenet-200'

    @property
    def tiny_imagenet_zip(self):
        return self.data / 'tiny-imagenet-200.zip'

# ------------------ CORRUPTED IMAGES ------------------

    @property
    def corruptions(self):
        return self.mkdir_if_not_exists(self.output / 'corruptions')
    
    def get_new_corruptions_files(self, corruption_name):
        original_path = self.corruptions / f'{corruption_name}_{datetime.datetime.now().strftime("_%Y_%m%d__%H_%M_%S")}_originals.pt'
        corruption_path = self.corruptions / f'{corruption_name}_{datetime.datetime.now().strftime("_%Y_%m%d__%H_%M_%S")}_corrupted.pt'
        labels_path = self.corruptions / f'{corruption_name}_{datetime.datetime.now().strftime("_%Y_%m%d__%H_%M_%S")}_labels.pt'
        plot_path = self.corruptions / f'{corruption_name}_{datetime.datetime.now().strftime("_%Y_%m%d__%H_%M_%S")}_plot.png'
        with original_path.open('w') as f:
            pass
        with corruption_path.open('w') as f:
            pass
        with labels_path.open('w') as f:
            pass
        with plot_path.open('w') as f:
            pass
        return original_path, corruption_path, labels_path, plot_path

project = CustomizedPath() 