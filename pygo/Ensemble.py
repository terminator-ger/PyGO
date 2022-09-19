import numpy as np
import pdb 
from abc import ABC, abstractmethod
 
 
class EnsembleMethod(ABC):
    def __init__(self, classifier):
        self.classifier = classifier

    @abstractmethod
    def predict(self, patches):
        pass


class SoftVoting(EnsembleMethod):
    def __init__(self, classifier):
        super().__init__(classifier)

    def predict(self, patches):
        prop = [cls.predict_prob(patches) for cls in self.classifier]
        prop = np.mean(prop, axis=0)
        val = np.argmax(prop, axis=1)
        return val


class MajorityVoting(EnsembleMethod):
    def __init__(self, classifier):
        super().__init__(classifier)

    def predict(self, patches):
        prop = [cls.predict(patches) for cls in self.classifier]
        #print('classifier 0')
        #print(prop[0].reshape(19,19))
        #print('classifier 1')
        #print(prop[1].reshape(19,19))
        #print('classifier 2')
        #print(prop[2].reshape(19,19))
        prop = np.stack(prop)

        hist = np.array([np.histogram(x, bins=6, range=(-0.5,5.5))[0] for x in prop.T])
        # major vote
        
        val = np.argmax(hist, axis=1)
        # conflicting (equal votes) -> empty board
        conflicts = np.argwhere(np.sum(hist==1, axis=1)==len(self.classifier))
        val[conflicts] = 2

        val[val==3] = 2
        val[val==4] = 2
        return val

