from abc import ABC, abstractmethod

class BaseGrader(ABC):
    @abstractmethod
    def grade(self, history: list) -> float:
        """
        Evaluate the score strictly between 0 and 1 (never exactly 0.0 or 1.0).
        """
        pass
