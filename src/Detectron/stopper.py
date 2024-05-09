class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.wait = 0
        assert mode in ['min', 'max']
        self.mode = mode

    def update(self, metric):
        if self.best is None:
            self.best = metric
            return False
        if self.mode == 'min':
            if metric < self.best - self.min_delta:
                self.best = metric
                self.wait = 0
                return False
            else:
                self.wait += 1
                return self.wait >= self.patience
        elif self.mode == 'max':
            if metric > self.best + self.min_delta:
                self.best = metric
                self.wait = 0
                return False
            else:
                self.wait += 1
                return self.wait >= self.patience
