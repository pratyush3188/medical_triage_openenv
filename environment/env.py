from .tasks.easy import EasyTask
from .tasks.medium import MediumTask
from .tasks.hard import HardTask
from .models import Action

class MedicalTriageEnv:
    def __init__(self):
        self.tasks = {
            "easy": EasyTask(),
            "medium": MediumTask(),
            "hard": HardTask()
        }
        self.current_task_name = "easy"
        self.current_task = self.tasks["easy"]

    def reset(self, task_name=None):
        if task_name and task_name in self.tasks:
            self.current_task_name = task_name
            self.current_task = self.tasks[task_name]
        return self.current_task.reset()

    def step(self, action: dict):
        action_obj = Action(**action)
        return self.current_task.step(action_obj)

    def state(self):
        return self.current_task.get_obs()
