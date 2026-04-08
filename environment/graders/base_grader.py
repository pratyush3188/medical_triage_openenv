from abc import ABC, abstractmethod

class BaseGrader(ABC):
    @abstractmethod
    def grade(self, history: list) -> float:
        """
        Takes the history of turns/actions and evaluates the score between 0.0 and 1.0
        """
        pass
