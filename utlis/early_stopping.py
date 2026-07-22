class EarlyStopping:


    def __init__(self,patience=10,mode="max",min_delta=0):

        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.counter = 0
        self.best_score = None
        self.early_stop = False


    def __call__(self, score):


        if self.best_score is None:

            self.best_score = score

            return False



        if self.mode == "max":
            improvement = (score > self.best_score + self.min_delta)


        else:
            improvement = (score < self.best_score - self.min_delta)


        if improvement:

            self.best_score = score
            self.counter = 0


        else:
            self.counter += 1

            if self.counter >= self.patience:

                self.early_stop = True



        return self.early_stop