# Documentation

## XOR Quick Example

### Getting training data and shape ready
First, decide what you are going to train the network to do.
In this example, the network will solve the XOR problem.

Our training data will look like this:
```
input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
output_data = [[0], [1], [1], [0]]
``` 
#### Shape
The shape of a neural network is denoted by a list of integers where each integer
determines how many nodes are on that corresponding layer. The first layer must
match the number of inputs and the last layer must match the number of outputs.

Shape is the only mandatory parameter in defining a species.

This network's shape must have 2 input nodes and 1 output node to match the
training data. Therefore, a shape of `[2, 1]` is the simplest network that could work.
Although, hidden layers are needed to solve XOR, so our shape will be `[2, 2, 1]`.

`shape = [2, 2, 1]`

### Creating the species
```
my_species = pygene.Species(shape, train_inputs=input_data, train_outputs=output_data, use_sigmoid=True, pop_size=1000, add_bias_nodes=True)                       
```

The `use_sigmoid` parameter is `True` because the sigmoid activation function is needed on
the hidden layers to solve XOR.

When the `add_bias_nodes` parameter is `True` an additional node is added to each
non-output layer.  These bias nodes take no incoming values and are always set to 1. While not needed for XOR, the bias nodes
can cause the network to converge sooner.

Both of those optional parameters are `True` by default as they tend to be helpful
more often than not.

### Training the species

`my_species.train(100)`

This will train the species for 100 generations.
To see how the network changes over time, break the training
up into many small segments and draw the best network 
between each segment:

```
for _ in range(100):
    my_species.train(1)
    pygame_window.blit(guide.get_best_network().draw(), (0, 0))
    pygame.display.update()
```

The `for` loop still trains it for 100 generations, but you can *see* how the network evolves.
If the networks are playing a game, you could show the best one playing between
training if you wanted.

### Using your trained network

`best_network = my_species.get_best_network()`

Collects the best network from the species

#### Calcio

`best_network.calico([0, 1])`

`calico` stands for *calculate output* and uses the provided inputs with the
network to get output(s).
With inputs `[0, 1]` trained to do XOR, we expect an output of nearly 1 from the network

`calico`'s optional parameter `show_internals` enables the value of all nodes
and edges to be printed to the console after the calculation. 
#### Hidden calico
Sometimes, you want to set the value of hidden nodes instead of just nodes on the 
input layer.

`best_network.calico_from_hidden_layer(1, [2, 3])`

Layer counting starts at 0 with the input layer, so layer 1 is the first hidden layer.
This line set the two nodes on layer 1 to the values 2 and 3 before they are activated, and returns
the output. All nodes before the layer being set are set to 0.
#### Visualizing

`best_network.draw(show_internals=True, independent=True)`

By setting `independent` to `True`, `draw` creates a pygame window with
the network drawing on it. The window must be closed to move on because the window is not run in parallel.
If you only want the pygame surface returned, set `independent` to `false`.

`show_internals` causes the values of all the nodes and edges to be drawn ontop of them.

![XORnetwork](https://github.com/Elan456/PyGene/blob/develop/PyGene/XORexample.png?raw=true)
The different colors matter. The dark blue color is for input nodes. The bright pink shows
the bias nodes. The blueish green is for hidden layers using a 
sigmoid activation function. Orange is for hidden layers without an activation function
Green is for the output nodes. 
## Full XOR Example Without Any Visualizations

```
import pygene

input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
output_data = [[0], [1], [1], [0]]

shape = [2, 2, 1]
my_species = pygene.Species(shape, 1, train_inputs=input_data, train_outputs=output_data,
                            use_sigmoid=True, pop_size=1000, add_bias_nodes=True)
my_species.train(100)
best_network = my_species.get_best_network()
print(best_network.calico([0, 1]))
```

This can easily be tinkered with by changing the training data and shape of the neural network.