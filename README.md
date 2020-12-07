# sprocket
A convolutional neural network that counts the number of mechanical chain links from an image. 
Built for the FIRST Robotics Competition (FRC).

uses tensorflow 2.3

## structure

### First Network

#### Input Data

2 image points, (x, y) (x, y) reorganized into (x1, y2, x3, y4) so that x1 < x2 and y1 < y2, regardless of correspondence.
