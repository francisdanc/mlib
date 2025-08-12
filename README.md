## MLib 

MLib is my own personal machine learning library. I've begun building this solely for learning purposes and plan to expand on it with several different machine learning algorithms.
Currently there are some errors in the xor example, the training converges at an MSE of around 0.25 indicating that it is not properly learning the input patterns. To improve this
I intend to add a bias to the network.

## Building a model
Building a model is quite simple in MLib the first step is to import the ncessary libraries:
```
import mlib.matrixes.InputMatrix;
import mlib.model.Model;
import mlib.network.layers.DenseLayer;
```
MLib uses the model class to modularly build models using the DenseLayer class. 

```
Model model = new Model();
DenseLayer h1 = new DenseLayer(2,2); // Create a denselayer that has an input size of 2 and 2 neurons
DenseLayer out = new DenseLayer(2,1); // create the output layer which has an input size of 2 and 1 neuron
model.addLayer(h1);
model.addLayer(out);
```

The main class used in MLibs neural networks is the InputMatrix class. The InputMatrix is used to create the matrices used by a model. 

```
// Create a new input matrix with a batch size of 4 and a feature length of 2
InputMatrix input = new InputMatrix(4, 2);
double[][] inputs = {{0.0,0.0}, {0.0,1.0}, {1.0,0.0}, {1.0,1.0}};
input.setInputMatrix(inputs); // Set the inputs

// Create the expected outputs matrix, batch size of 4 with a feature length of 1
InputMatrix expected = new InputMatrix(4,1);
double[][] e = {{1.0}, {0.0}, {0.0}, {1.0}};
expected.setInputMatrix(e); // Set the expected matrix
```
This is all of the setup necessary in order to make a simple MLP which can predict the outcome of an XOR logic gate. To train the model, you use the model object created earlier.
```
// Train a model for 10,000 epochs with a learning rate of 0.01 using the input and expected matrix
model.train(input, 0.01, 10000, expected); 
```
