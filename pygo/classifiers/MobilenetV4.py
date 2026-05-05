import os
import torch as th
import numpy as np

from pygo.utils.data import load_and_augment_training_data, weights_path
from pygo.GoNet import GoNet
from pygo.classifiers.BaseGoClassifier import Classifier
from timm import create_model

class MobilenetV4Classifier(Classifier):
    _parameter_constraints = {
        "weights_file": [str],
        "num_classes": [int],
    }
     
    def __init__(self, weights_file, classes=3) -> None:
        self.classes_ = classes
        self.weights_file = weights_file
        self._is_fitted = True
            
    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    def predict(self, patches):
        if not hasattr(self, 'model'):
            self.model = create_model('mobilenetv4_conv_small.e1200_r224_in1k', pretrained=True, num_classes=3, )
            self.model.eval()
            self.load()

        x = th.from_numpy(np.array(patches).astype(np.float32)).permute(0,3,1,2)
        if th.max(x) > 1.0:
            x = x / 255
        lbl = self.model(x)
        lbl = lbl.detach().cpu().numpy()
        lbl = np.argmax(lbl, axis=1)

        for size in [9, 13, 19]:
            if len(patches) == size*size:
                lbl = lbl.reshape(size, size, self.classes_)
                lbl = np.rot90(np.fliplr(lbl))
                break
 
        lbl = lbl.reshape(-1)
        return lbl


    def predict_proba(self, patches):
        if not hasattr(self, 'model'):
            self.model = create_model('mobilenetv4_conv_small.e1200_r224_in1k', pretrained=True, num_classes=3, )
            self.model.eval()
            self.load()


        x = th.from_numpy(np.array(patches).astype(np.float32)).permute(0,3,1,2)
        if th.max(x) > 1.0:
            x = x / 255
        lbl = self.model(x)
        lbl = lbl.detach().cpu().numpy()
        
        for size in [9, 13, 19]:
            if len(patches) == size*size:
                lbl = lbl.reshape(size, size, self.classes_)
                lbl = np.rot90(np.fliplr(lbl))
                break
 
        lbl = lbl.reshape(-1, self.classes_)
 
        return lbl



    def fit(self):
        raise NotImplementedError("Training not implemented for CnnClassifier")
        self.model = GoNet(num_classes=self.num_classes)
        X_train, y_train, X_test, y_test = load_and_augment_training_data((lambda x:x))
       
        X_train = th.from_numpy(X_train.astype(np.float32))
        y_train = th.from_numpy(y_train.astype(np.int_))
        X_test  = th.from_numpy(X_test.astype(np.float32))
        y_test  = th.from_numpy(y_test.astype(np.int_))
        # channels to pos 1
        X_train = X_train.permute(0,3,1,2)
        X_test  = X_test.permute(0,3,1,2)

        batch_size = X_train.size()[0]
        opt = th.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.005)
        loss_fn = th.nn.CrossEntropyLoss()
        print('train')
        for i in range(40):
            permutation = np.arange(X_train.size()[0])
            
            for j in range(0, X_train.size()[0], batch_size):
                indices = permutation[j:j+batch_size]
                batch_x, batch_y = X_train[indices], y_train[indices]
                y_pred = self.model(batch_x)
                loss = loss_fn(y_pred, batch_y)
                loss.backward()
                opt.step()
                opt.zero_grad()

            with th.no_grad():
                y_pred = self.model(X_test)
                loss_test = loss_fn(y_pred, y_test)
                y_pred = F.log_softmax(y_pred, -1)
                y_pred = toNP(y_pred)
                y_pred = np.argmax(y_pred, axis=1)
                f1 = f1_score(y_test, y_pred, average='micro')
                print("Epoch {} : {:0.2f}, {:0.3f}, {:0.3f}".format(i,f1, toNP(loss), toNP(loss_test)))

            if i % 5 == 0:
                th.save(self.model, 'weights_{}_{}.pt'.format(i, f1))
                for cls in range(self.num_classes):
                    plots=101+self.num_classes*10
                    plt.subplot(plots+cls)
                    if np.any((y_pred==cls)):
                        plt.imshow(np.vstack(X_test[y_pred==cls, 0]))

                plt.savefig('{}.png'.format(i), dpi=400)

        print(classification_report(y_test, y_pred))
        self.hasWeights = True

    def load(self):
        weights_file = weights_path("weights", self.weights_file)
        if os.path.exists(weights_file):
            self.model.load_state_dict(th.load(weights_file, weights_only=True))
        else:
            print('Failed to Restore ConvGO Classification Alg')

    def store(self):
        weights_file = weights_path("weights", self.weights_file)
        th.save(self.model, weights_file)