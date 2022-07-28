# STAT-5241-Statistical-Machine-Learning-Final-Project

## Project Summary:

This project aims to train deep learning models to handle [The MNIST handwritten digit classification problem](http://yann.lecun.com/exdb/mnist/). In this project, I first tried out different simple machine learning algorithms like KNN, logistic regression, SVM to see whether I can get some decent result. Latter, I applied naive nerual network to see the improvement of classification. To further improve the classification accuracy, I applied convolutional neural network along with various optimization techniques such as batch normalization and dropout. The Final test accuracy achieves 98.9% (i.e. 1.1% test error).

## My Model:
My final model is shown in the following:

Model: Convolutional Nerual Network (CNN)

Network structure (using Pytorch):

    # 2 convolutional layer with batch normalization
    # a Max pooling layer
    # Dropout applied

    def __init__(self, n_hidden, n_output):
        super(OurCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5)
        self.conv1_bn = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(16, 33, 5)
        self.conv2_bn=nn.BatchNorm2d(33)

        self.dropout1=nn.Dropout(0.25)

        self.fc1 = torch.nn.Linear(33*4*4 , hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
   
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(self.conv1_bn(x)))
        x = self.conv2(x)
        x = self.pool(F.relu(self.conv2_bn(x)))
        x = self.dropout1(x)
        x = x.view(-1, 33*4*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
       
