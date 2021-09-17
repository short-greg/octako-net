
import numpy as np
import tqdm


class Visual(object):

    def __init__(
        self, progress: status.TrainingProgress, performance: status.Performance
    ):
        self.progress = progress
        self.performance = performance

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, tb):
        return
    
    def update(self, progress, performance):
        raise NotImplementedError
    
    def segment(self):
        NotImplementedError


class NullVisual(Visual):

    def __init__(
        self, progress: status.TrainingProgress, performance: status.Performance
    ):
        super().__init__(progress, performance)

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, tb):
        return
    
    def update(self, progress: status.TrainingProgress, performance: status.Performance):
        # add in the update function
        # progress.n_epochs
        self.progress = progress
        self.performance = performance

    def segment(self):
        return


class TQDMVisual(Visual):

    def __init__(
        self, progress: status.TrainingProgress, performance: status.Performance
    ):
        super().__init__(progress, performance)
        self.pbar = None

    def __enter__(self):
        self.pbar = tqdm.tqdm(total=self.progress.n_iterations)
        return self
    
    def __exit__(self, type, value, tb):
        self.pbar.close()
        return
    
    def update(self, progress: status.TrainingProgress, performance: status.Performance):
        # add in the update function
        # progress.n_epochs
        self.progress = progress
        self.performance = performance
        self.pbar.total = progress.n_iterations
        self.pbar.update(1)
        self.pbar.refresh()
        self.pbar.set_postfix(dict(
            **{
                "State": progress.state_name
            },
            **performance.to_average_dict()
        ))

    def segment(self):
        print()


class Visualizer(object):

    def __init__(self, visual_cls):
        """
        Args:
            visual_cls 
        """
        self.visual_cls = visual_cls

    def visualize(self, progress, performance):
        return self.visual_cls(progress, performance)
