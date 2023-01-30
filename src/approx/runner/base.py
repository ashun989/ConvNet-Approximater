from abc import ABCMeta, abstractmethod


class BaseRunner(metaclass=ABCMeta):

    @abstractmethod
    def run(self):
        pass
