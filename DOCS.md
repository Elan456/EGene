# PyGene Documentation

## XOR Example

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

The optional parameter `show_pop` determines if the program prints the average loss and all losses of the population.
Useful for seeing how homogenous the population is.

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
Both  
`best_network = my_species.get_best_network()`  
and  
`best_network = my_species.networks[0]`  
collect the best network from the species because the network are sorted from best to worst.

#### Calico

`best_network.calico([0, 1])`

`calico` stands for *calculate output* and uses the provided inputs with the
network to get output(s).
With inputs `[0, 1]` trained to do XOR, we expect an output of nearly 1 from the network.

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

`show_internals` causes the values of all the nodes and edges to be drawn on top of them.

![XORnetwork](https://github.com/Elan456/PyGene/blob/develop/PyGene/XORexample.png?raw=true)

The different colors matter:

**Nodes**  
Dark Blue: Input nodes  
Blueish green: Sigmoid activated hidden node  
Orange: Hidden node without activation function  
Pink: Bias node  
Green: Output node

**Edges**  
Green: Positive values  
Red: Negative values
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

---

## All Species Parameters

- `shape`:    
        A list of values where each value is the number of nodes in that layer. The first value 
        must be the number of inputs and the last value is the number of outputs. For example, a shape of [2,4,1,3] would have
        2 inputs, 3 outputs, and a total of 5 hidden nodes with 4 on the second layer and 1 on the third layer.


- `initial_change_rate`:  
        The initial change rate used when mutating the networks. If can_change_rate is True then, the change rate will be
        halved everytime there is no improvement after 4 generations.


- `pop_size`:  
        The number of individuals in the species. Values lower than 32 are not recommended because when the population is
        divided by sixteenths, there would be less than 2 individuals in a section.


- `train_inputs` + `train_outputs`:  
        The inputs and outputs used when there is no custom score function. The length train_inputs must match the length of
        train_outputs. Because there can be multiple inputs in a network, each example must be in a nested list.
        For example, inputs of [[2],[3],[6]] would need a network with 1 input node whereas an input set of
        [[2,3,6], [4,2,7]] would need a network with 3 inputs nodes. The first input correlates to the first output and so
        on.


- `loss_function`:  
        Pass a function. The function must take a network as its first parameter and return the amount of loss. If you want
        to use score instead of loss, just multiply the score by -1 to get the loss so higher scores are a lower loss.
        Within the function you will want to use the network.calico method to pass in inputs and get outputs. Those outputs
        could control a video game character or anything, so you can figure out how good the network is.


- `initial_weights`:  
        A list of weights that the first network will be given. Useful when training needs to be paused and then resumed
        without losing all progress.


- `data_per_gen`:  
        When not using a custom score function, determines how many of the input/output sets given are used in each
        generation. Useful when massive amounts of training data are given, but they shouldn't all be used each generation.
        Each generation, a random selection of training data is used with data_per_gen size. Can't be larger than the number
        of input sets being given.


- `use_sigmoid`:  
        Determines if the hidden nodes use a sigmoid activation function. If false, no activation functions will be used.


- `can_change_changerate`:  
        Whether the program will halve the change_rate when there is no change in loss after 4 generation.


- `use_multiprocessing`:  
        Only used for custom score functions. Lets the networks be scored in parallel, utilizing more cpu.


- `set_all_zero`:  
       Initializes all networks with weights of zero instead of random values


- `add_bias_nodes`:  
       When true, a bias node (drawn with a pink color) is added to every non-output layer and always has a value of 1.


- `native_window_size`:   
    Is both the width and height of the pygame surface when a network is drawn.
