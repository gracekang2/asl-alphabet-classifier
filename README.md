### Static ASL Alphabet Translation w/ Deep Learning

<img src="https://github.com/gracekang2/asl-alphabet-classifier/blob/master/DEMO.gif" alt="DEMO" width="400" />

Uses OpenCV, deep learning models built with fastai and PyTorch. The actual live feed translation is in asl_classifier.py.

I tried three different approaches to training a model, all with only the **static ASL alphabet signs** (i.e. excluding Z and J):

1. Fine-tuning resnet34, with fastai, with a self-generated dataset.

   <img src="https://github.com/gracekang2/asl-alphabet-classifier/blob/master/loss_1.png" alt="loss_1" width="400" />

   <img src="https://github.com/gracekang2/asl-alphabet-classifier/blob/master/confusion_1.png" alt="confusion_1" width="400" />

2. Training a CNN from scratch using PyTorch with this MNIST ASL alphabet dataset from Kaggle: https://www.kaggle.com/datamunge/sign-language-mnist 

   - I tried with this MNIST dataset before moving to the custom dataset, since this MNIST dataset contains small 28x28 images. I wanted to try on smaller images to make sure the CNN was training properly before moving to higher resolution pictures.

   - Trained for 10 epochs with 3 different optimizers, best accuracy was 95%.

   - Confusion matrix:

     <img src="https://github.com/gracekang2/asl-alphabet-classifier/blob/master/confusion_2.png" alt="confusion_2" width="400" />

3. Training a CNN from scratch using PyTorch with the self-generated dataset.

   - Trained for 10 epochs, accuracy was 97%.

   - Confusion matrix:

     <img src="https://github.com/gracekang2/asl-alphabet-classifier/blob/master/confusion_3.png" alt="confusion_3" width="400" />

Overall, the model with the best outcome was #1, in terms of accuracy while training and subjective testing with the live feed. All of the training code can be found in train_model.ipynb. 



