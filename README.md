#Distillation as a Defense to Adversarial Perturbations 
#against Deep Neural Networks
#Haobei Song, Mahesh Tripunitara(13 March 2017)
#
##There are three CNN training and testing scripts with the same architecture
#InputLayer
#Relu Convolutional 32 3X3 filters
#Relu Convolutional 32 3X3 filters
#Max Pooling 2X2
#Relu Convolutional 64 3X3 filters
#Relu Convolutional 64 3X3 filters
#Max Pooling 2X2
#Relu Fully Connect 200 units
#Relu Fully Connect 200 units
#Softmax 10 units
#
#Learing Rate 0.1
#Momentum 0.5
#Dropuout Rate (Fully Connect) 0.5
#Batch size 128 
#Epochs (Default:50)
#Temperature (Default:20)
#Wrong test sample numbers per batch (Default:0)

#The standard CNN with Temperature at 1 
./std_CNN.py
#The pre-training with Epochs=E, Sample Number=S, Temperature at T is called as
./pre_CNN.py E S T
#The post-training is called the same as above
./post_CNN.py E S T

#the test samples for wrong tests for std_CNN are stored in 
saved_pics_old
#the test samples for wrong tests for pre_CNN are stored in 
saved_pics_init
#the test samples for wrong tests for post_CNN are stored in 
saved_pics

#The distilled labels are produced automatically after the call to pre_CNN.py and the resulting file is saved as
distilled_labels.npz

#Reference: 
#1.Lasagne Documentation
#2.N.Papernot, P.McDaniel, et al, "Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks," in Symposium on Security & Privacy, IEEE 2016. San Jose, CA.

