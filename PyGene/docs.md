# Documentaion

## XOR Quick Example

### Getting training data and shape ready
First, decide what you are going to train the network to do.
In this example, the network will solve the XOR problem.

Our training data will look like this:
```
input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
output_data = [[0], [1], [1], [0]]
``` 
This network's shape must have 2 input nodes and 1 output node to match the
training data. Therefore, a shape of `[2, 1]` is the simplest network that could work.
Although, hidden layers are needed to solve XOR, so our shape will be `[2, 2, 1]`.

`shape = [2, 2, 1]`

### Creating the species
```
my_species = pygene.Species(shape, 1, train_inputs=input_data, train_outputs=output_data, use_sigmoid=True, pop_size=1000, add_bias_nodes=True)                       
```

The `use_sigmoid` parameter is `True` because the sigmoid activation function is needed on
the hidden layers to solve XOR

The `add_bias_nodes` parameter is `True` to add an additional node to each
non-output layer that is always equal to 1. While not needed, the bias nodes
cause the network to converge sooner.

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

#### Visualize

`best_network.draw(show_internals=True, independent=True)`

Creates a window showing the network instead of
just returning a surface because `independent` is `True`. 
The window must be closed to move on because the window is not run in parallel.

![XORnetwork](https://github.com/Elan456/PyGene/blob/develop/PyGene/XORexample.png?raw=true)

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
best_network.calico([0, 1])
```