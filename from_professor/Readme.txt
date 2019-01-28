

[Dec. 12, 2018]
CNN model architectures

input (99x257)
output (16)

a. 1Conv - 1FC 
conv1(5x5; 8)
pool1(2x2)
fc1(1024) 

b. 1Conv - 2FC
conv1(5x5; 8)
pool1(2x2) 
fc1(1024) 
fc2(512)

c. 2Conv - 1FC
conv1(5x5; 8)
pool1(2x2)
conv2(5x5; 16)
pool2(2x2)
fc1(1024) 

d. 2Conv - 2FC
conv1(5x5; 8)
pool1(2x2)
conv2(5x5; 16)
pool2(2x2)
fc1(1024)
fc2(512)

e. 3Conv - 1FC
conv1(5x5; 8)
pool1(2x2)
conv2(5x5; 16)
pool2(2x2)
conv3(5x5; 32)
pool3(2x2)
fc1(1024) 

f. 3Conv - 2FC
conv1(5x5; 8)
pool1(2x2) 
conv2(5x5; 16)
pool2(2x2)
conv3(5x5; 32)
pool3(2x2)
fc1(1024)
fc2(512) 

g. 4Conv - 1FC
conv1(5x5; 8)
pool1(2x2)
conv2(5x5; 16)
pool2(2x2)
conv3(5x5; 32)
pool3(2x2)
conv4(5x5; 64)
pool4(2x2)
fc1(1024) 

h. 4Conv - 2FC
conv1(5x5; 8)
pool1(2x2) 
conv2(5x5; 16)
pool2(2x2)
conv3(5x5; 32)
pool3(2x2)
conv4(5x5; 64)
pool4(2x2)
fc1(1024)
fc2(512) 

i. 5Conv - 1FC
conv1(5x5; 8)
pool1(2x2)
conv2(5x5; 16)
pool2(2x2)
conv3(5x5; 32)
pool3(2x2)
conv4(5x5; 64)
pool4(2x2)
conv5(5x5; 128)
pool5(2x2)
fc1(1024) 

j. 5Conv - 2FC
conv1(5x5; 8)
pool1(2x2) 
conv2(5x5; 16)
pool2(2x2)
conv3(5x5; 32)
pool3(2x2)
conv4(5x5; 64)
pool4(2x2)
conv5(5x5; 128)
pool5(2x2)
fc1(1024)
fc2(512) 

k. 6Conv - 1FC
conv1(2x3; 8)
pool1(2x2)
conv2(2x3; 16)
pool2(2x2)
conv3(2x3; 32)
pool3(2x2)
conv4(2x3; 64)
pool4(2x2)
conv5(2x3; 128)
pool5(2x2)
conv6(2x3; 256)
pool6(2x2)
fc1(1024) 

l. 6Conv - 2FC
conv1(2x3; 8)
pool1(2x2)
conv2(2x3; 16)
pool2(2x2)
conv3(2x3; 32)
pool3(2x2)
conv4(2x3; 64)
pool4(2x2)
conv5(2x3; 128)
pool5(2x2)
conv6(2x3; 256)
pool6(2x2)
fc1(1024) 
fc2(512) 









a. 2 Conv - 2 FC 

1 FC (1024)
2 FC layers (1024-512)
conv1 (5x5) - pool1 (2x2) - 

b. 3 Convolutional layers
1 FC layer (1024)
2 FC layer (1024-512)

c. 5 Convolutional layers
1 FC layer (1024)
2 FC layers (1024-512)

d.


