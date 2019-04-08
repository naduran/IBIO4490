#!/home/afromero/anaconda3/bin/ipython
# read kaggle facial expression recognition challenge dataset (fer2013.csv)
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
import numpy as np
import matplotlib.pyplot as plt
import os

directory=os.listdir(os.getcwd())

if "fer2013.csv" not in directory:
  os.system('wget http://bcv001.uniandes.edu.co/fer2013.zip')
  os.system('unzip fer2013.zip')

def sigmoid(x):
    return 1/(1+np.exp(-x))

def get_data():
    # angry, disgust, fear, happy, sad, surprise, neutral
    with open("fer2013.csv") as f:
        content = f.readlines()

    lines = np.array(content)
    num_of_instances = lines.size
    print("number of instances: ",num_of_instances)
    print("instance length: ",len(lines[1].split(",")[1].split(" ")))

    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(1,num_of_instances):
        emotion, img, usage = lines[i].split(",")
        pixels = np.array(img.split(" "), 'float32')
     
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)

    #------------------------------
    #data transformation for train and test sets
    x_train = np.array(x_train, 'float64')
    y_train = np.array(y_train, 'float64')
    x_test = np.array(x_test, 'float64')
    y_test = np.array(y_test, 'float64')

#    x_train /= 255 #normalize inputs between [0, 1]
#    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], 48, 48)
    x_test = x_test.reshape(x_test.shape[0], 48, 48)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # plt.hist(y_train, max(y_train)+1); plt.show()

    return x_train, y_train, x_test, y_test

class Model():
    def __init__(self):
        params = 48*48 # image reshape
        out = 7 
        self.lr = 0.001 # Change if you want
        self.W = np.random.randn(params, out)
        self.b = np.random.randn(out)

    def forward(self, image):
        image = image.reshape(image.shape[0], -1)
        out = np.dot(image, self.W) + self.b
        return out

    def compute_loss(self, pred, gt):
        J = (-1/pred.shape[0]) * np.sum(np.multiply(gt, np.log(sigmoid(pred))) + np.multiply((1-gt), np.log(1 - sigmoid(pred))))
        return J

    def compute_gradient(self, image, pred, gt):
        image = image.reshape(image.shape[0], -1)
        W_grad = np.dot(image.T, pred-gt)/image.shape[0]
        self.W -= W_grad*self.lr

        b_grad = np.sum(pred-gt)/image.shape[0]
        self.b -= b_grad*self.lr

def train(model):
    x_train, y_train, x_test, y_test = get_data()
    batch_size = 100 # Change if you want
    epochs = 40000 # Change if you want
    ep=[];
    l_train=[];
    l_test=[];
    for i in range(epochs):
        loss = []
        for j in range(0,x_train.shape[0], batch_size):
            _x_train = x_train[j:j+batch_size]
            _y_train = y_train[j:j+batch_size]
            out = model.forward(_x_train[0:int(x_train.shape[0]/2)])
            loss.append(model.compute_loss(out, _y_train[0:int(y_train.shape[0]/2)]))
            model.compute_gradient(_x_train[0:int(x_train.shape[0]/2)], out, _y_train[0:int(y_train.shape[0]/2)])
        out = model.forward(x_train[int(x_train.shape[0]/2):x_train.shape[0]])                
        loss_test = model.compute_loss(out, y_train[int(x_train.shape[0]/2):x_train.shape[0]])
        print('Epoch {:6d}: {:.5f} | test: {:.5f}'.format(i, np.array(loss).mean(), loss_test))
        ep.append(i)
        l_train.append(np.array(loss).mean())
        l_test.append(loss_test)
        plot(ep,l_train,l_test)

def plot(ep,loss_train,loss_test): # Add arguments
    pdf=plt.figure()
    plt.plot(ep,loss_train,'r',ep,loss_test,'g')
    plt.xlabel('Prediction Error')
    plt.ylabel('Model Complexity')
    plt.legend('Train','Test')
    pdf.savefig("40kepoch_100batch.pdf")
    # CODE HERE
    # Save a pdf figure with train and test losses
    pass

def test(model):
    _, _, x_test, y_test = get_data()
    
    # YOU CODE HERE
    # Show some qualitative results and the total accuracy for the whole test set
    pass

if __name__ == '__main__':
    model = Model()
    train(model)
    test(model)

