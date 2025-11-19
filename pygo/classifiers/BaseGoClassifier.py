from sklearn.base import BaseEstimator, ClassifierMixin

class Classifier(ClassifierMixin, BaseEstimator):
    def predict(self, patches):
        raise NotImplementedError()

    def train(self, patches):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def store(self):
        raise NotImplementedError()

