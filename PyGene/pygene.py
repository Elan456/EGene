import copy
import random
import math
import pygameTools as pgt

import time
from multiprocessing import Pool

import pygame
from pygame import gfxdraw

pygame.init()

# else:
gameDisplay = pygame.Surface((10, 10))

black = (0, 0, 0)
white = (255, 255, 255)

colors = {"input": (37, 37, 125),
          "hidden": (100, 0, 0),
          "output": (0, 150, 0),
          "bias": (200, 0, 200)}


def breaklist(L, list_count):
    lists = []
    items_per_list = len(L) / list_count

    for v in range(list_count):
        lists.append([])

    for v in range(len(L)):
        lists[int(v / items_per_list)].append(L[v])
    return lists


def sigmoid(x):
    # print("USING SIGMOID")
    try:
        y = 1 / (1 + math.e ** (-1 * x))
    except OverflowError:
        y = 0
    return y


def donata(x):
    return x


def square(x):
    return x ** 2


class CustomError(Exception):
    pass


def duplicatechecker(x):
    uniquevalues = []

    for v in x:
        if v not in uniquevalues:
            uniquevalues.append(v)
        else:
            return True
    return False


def custom_eval(t):
    scorefunction, network = t
    return scorefunction(network)


class Species:
    def __init__(self, shape, changerate, popsize=32, train_inputs=None, train_outputs=None, scorefunction=None,
                 initial_weights=None, datapergen=None, use_sigmoid=True, can_change_changerate=True,
                 use_multiprocessing=True, set_all_zero=False, add_bias_nodes=True):
        self.use_multiprocessing = use_multiprocessing
        self.use_sigmoid = use_sigmoid
        self.can_change_changerate = can_change_changerate
        self.set_all_zero = set_all_zero
        self.add_bias_nodes = add_bias_nodes

        self.epochs = 0  # Count of all epochs every trained on this species
        self.lowestlost = float("inf")
        self.all_blost = []  # All the lowest losses for each epoch

        if scorefunction == None and train_inputs == None:
            raise CustomError("Species needs either a list of inputs and outputs or a scorefunction")
        if scorefunction == None:
            self.n_inputs = len(train_inputs[0])
            self.n_outputs = len(train_outputs[0])
            self.scorefunction = self.evaluate
            self.using_custom_score_function = False

        else:
            self.scorefunction = scorefunction
            self.using_custom_score_function = True

        self.shape = shape  # This does not include the bias nodes which are added to every non-output layer

        if not self.using_custom_score_function:
            if self.shape[0] != self.n_inputs:
                raise CustomError("First layer node count does not equal inputs number from training data which is:",
                                  self.n_inputs)
            if self.shape[-1] != self.n_outputs:
                raise CustomError("Last layer node count does not equal outputs number from training data which is:",
                                  self.n_outputs)
            if duplicatechecker(train_inputs):
                print("--Duplicate Inputs Found--")
            if datapergen is None:
                datapergen = len(train_inputs)
            self.datapergen = min(len(train_inputs), datapergen)
        else:
            # print("Datepergen set to None because self.using_custom... is", self.using_custom_score_function)
            self.datapergen = None

        self.initalweights = initial_weights

        # print("datagennum:", self.datapergen)

        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.popsize = popsize
        self.changerate = changerate

        self.networks = []

        for p in range(popsize):  # Creating first generation of networks
            self.networks.append(Network(self.shape, self.use_sigmoid, self.add_bias_nodes, self.set_all_zero))
        print("--First networks made")
        # INITAL WEIGHTS
        if self.initalweights is not None and self.set_all_zero is False:
            for v in range(len(self.networks[0].w)):
                self.networks[0].w[v].value = self.initalweights[v]
        # print("-- Initial Weights initalized")
        # for p in self.networks:
        #     print(p.calico([2, 3]))

    @staticmethod
    def evaluate(p, inputs, output):  # Using an input it calculates the loss or score of the network default scorefuction
        loss = 0

        # print("initalloss:",self.loss)
        # print("input:",input)

        for I in range(len(inputs)):

            gen_output = p.calico(inputs[I])
            desiredoutput = output[I]
            # print("outputdesired:",desiredoutput)
            # print("genout:", gen_output, "realout", desiredoutput)
            for o in range(len(gen_output)):  # for handeling networks with multiple outputs
                loss += abs(gen_output[o] - desiredoutput[o])
                # print("thereloss:", abs(gen_output[o] - desiredoutput[o]))
        # print("predivideloss:", self.loss)

        loss /= len(inputs)  # division for average loss
        # print(self.loss)
        # print("Desired Output:", output, "Output reciceved", gen_output, "loss", self.loss)
        return loss

    def scoreall(self, show, scorefunction):  # Evaluates all of the networks and puts them in order from best to worst

        if self.using_custom_score_function:  # if a custom function is being use
            if self.use_multiprocessing:
                data = []
                p = Pool()
                for a in self.networks:
                    data.append((self.scorefunction, a))

                results = p.map(custom_eval, data)

                for a in range(len(self.networks)):
                    self.networks[a].loss = results[a]
            else:
                for a in range(len(self.networks)):
                    self.networks[a].loss = self.scorefunction(self.networks[a])
        else:  # If a list of inputs and outputs are being used for training
            inout = []  # List of tuples with each tuple having input, output
            for i in range(len(self.train_inputs)):
                inout.append((self.train_inputs[i], self.train_outputs[i]))
            random.shuffle(inout)
            ins = []
            outs = []
            for i in range(self.datapergen):
                ins.append(inout[i][0])
                outs.append(inout[i][1])
            for p in self.networks:
                p.loss = scorefunction(p, ins, outs)

        self.networks.sort(key=lambda x: x.loss)

    def crossover(self, p1, p2):  # Crosses the weights of two parents to get a new child
        child = Network(self.shape, self.use_sigmoid, self.add_bias_nodes)
        num_weights = len(p1.w)
        cross_point = random.randint(0, num_weights - 1)
        orientation = random.choice([(p1, p2), (p2, p1)])  # Determines which parent is first
        # print()
        # print("Parent 1:", [w.value for w in p1.w])
        # print("Parent 2:", [w.value for w in p2.w])
        # print("Cross point:", cross_point)

        for v in range(len(child.w)):  # Changes every weight
            if v < cross_point:
                child.w[v].value = orientation[0].w[v].value
            else:
                child.w[v].value = orientation[1].w[v].value

        # print("Child   :", [w.value for w in child.w])
        # print()

        return child

    def mutate(self, network):  # Adds some random variation to some weights
        # print("Before mutation:", [w.value for w in network.w])
        for w in network.w:
            w.value += random.random() * random.choice([-1, 0, 0, 0, 1]) * self.changerate
        # print("After  mutation:", [w.value for w in network.w])
        return network

    def nextgen(self):  # Crosses over and mutates certain networks
        choices = []
        # print("NEWGEN")
        # print("\n\n")

        # print(choices)
        best_weights = self.networks[0].w
        n = 0
        # print()
        # print("ENTIRE POPULATION BEFORE CHANGES")
        # for p in self.networks:
        #     print([w.value for w in p.w])
        max_index = max(4, int(self.popsize / 16))  # The worst network possible for reproduction

        for p in range(len(self.networks) // 16, len(self.networks)):  # Only the worst 15/16 are changed

            n += 1
            while True:  # Choosing the two parents
                # print("0 to", max(2, int(self.popsize / 16)))

                p1 = self.networks[random.randint(0, max_index)]
                p2 = self.networks[random.randint(0, max_index)]
                # print(p1,p2)

                if p1 != p2:
                    break
            # print("p1,p2", p1, p2)

            self.networks[p] = self.crossover(p1, p2)  # Crosses the parents to produce a child
            self.networks[p] = self.mutate(self.networks[p])  # Mutates the child based on the change rate
        # print("ENTIRE POPULATION AFTER CHANGES")
        # for p in self.networks:
        #     print([w.value for w in p.w])

        # print("Changed:",n)
        # print("Newval:", p.w[j].value)
        # print(int(len(self.networks)*(15/16)))

        for p in range(int(len(self.networks) * (15 / 16)),
                       len(self.networks)):  # Last 16th are just asexual mutants of the best 16th only one weight is changed
            p1 = self.networks[random.randint(0, max_index)]
            w_index = random.randint(0, len(self.networks[0].w) - 1)
            # for i, v in enumerate(self.networks[p].w):
            self.networks[p].w = copy.deepcopy(p1.w)
            self.networks[p].w[w_index].value = random.choice([-1, 1]) * random.random() * self.changerate + p1.w[
                w_index].value

    def train(self, epochs, show_pop=False):

        for v in range(epochs):

            self.scoreall(True, self.scorefunction)

            self.all_blost.append(self.networks[0].loss)
            print("\n", self.epochs, ":", "loss:", self.networks[0].loss, self.networks[0].show())
            # print("-2:", all_blost[v-2], "v:", all_blost[v])
            if len(self.all_blost) > 4 and self.all_blost[self.epochs - 4] == self.all_blost[
                self.epochs] and self.can_change_changerate:
                self.changerate /= 2
                print("--\nNo change from 4 gens ago so changerate is being lowered to", self.changerate, "\n--")

            if show_pop:
                all_losses = [a.loss for a in self.networks]
                print("Avg:", sum(all_losses) / len(all_losses), "Losses:", all_losses)
            self.epochs += 1
            self.nextgen()

    def get_best_network(self):
        return self.networks[0]


class Network:
    def __init__(self, shape, use_sigmoid, add_bias_nodes, set_all_zero=False):
        self.loss = 0
        self.use_sigmoid = use_sigmoid
        self.set_all_zero = set_all_zero
        self.shape = shape

        # Initiate nodes ------------
        self.nodes = []
        xscale = 500 / (len(self.shape) + 1)

        xstart = xscale
        layer_starts = []  # Keeps track of where each layer starts so weights are created faster

        # Determining the radius of the nodes when drawn so they all fit
        self.node_draw_size = min(int(450 / max(self.shape) / 2.2), int(xscale / 10))
        # print("creating nodes")

        # Node Creation
        for l in range(len(self.shape)):  # Each layer
            layer_starts.append(len(self.nodes))
            x = int(l * xscale + xstart)

            if add_bias_nodes and l != len(self.shape) - 1:  # Prevents bias on output layer
                layersize = self.shape[l] + 1  # +1 for bias
            else:
                layersize = self.shape[l]
            for n in range(layersize):  # Each node in the layer +1 for bias:
                if not (n == self.shape[l] and l == len(self.shape) - 1):  # Prevents bias on output layer
                    yscale = 500 / (layersize + 1)
                    y = int(n * yscale + yscale)
                    if l == 0:
                        type = "input"
                    elif l == len(self.shape) - 1:
                        type = "output"
                    else:
                        type = "hidden"

                    if n == self.shape[l]:
                        type = "bias"  # Overrides other types
                    self.nodes.append(Network.Node(type, (x, y), l, n, self.use_sigmoid, self.node_draw_size))

        layer_starts.append(len(self.nodes))
        # print("nodes made")
        # print([n.layer for n in self.nodes])
        # print("layer_Starts:", layer_starts)

        # Initate weights
        # print("creating weigths")
        timer = time.time()

        # Initalizing all the weights
        things_checked = 0
        self.w = []
        for n in self.nodes:
            if n.layer != len(layer_starts) - 2:
                # print(layer_starts[n.layer+1],"to", layer_starts[n.layer+2])
                for target_index in range(layer_starts[n.layer + 1],
                                          layer_starts[n.layer + 2]):  # All nodes ahead in index are checked
                    things_checked += 1
                    t = self.nodes[target_index]

                    if t.layer == n.layer + 1:  # If the target node is one layer ahead of the current node
                        # print("tnode is:", "layer:", t.layer, "node:", t.node)

                        if t.node < self.shape[t.layer]:  # Stops weights from connecting to the bias node,
                            # weights can only connect from the bias node not to bias node.
                            #   print("Good to conncet to")
                            if self.set_all_zero is False:
                                self.w.append(Network.Edge(random.choice([-1, 1]) * random.random(), n, t))
                            else:
                                self.w.append(Network.Edge(0, n, t))
                        else:
                            pass
                        #  print("A bias node")
        # print("weights made. T_C:", things_checked,"Time: ", time.time() - timer, "# of weights created:", len(self.w))
        self.w.sort(key=lambda x: x.pnode.layer)

        # print(self.shape)
        # for v in self.w:
        #     print([v.pnode.layer, v.pnode.node],[v.tnode.layer, v.tnode.node])

    def set_weights(self, weights):
        for v in range(len(self.w)):
            self.w[v].value = weights[v]

    def set_input_nodes_to_input_values(self, inputs):
        for n in self.nodes:
            n.value = 0

            if n.layer == 0:  # If it is on the input layer
                if n.node == self.shape[0]:  # Checks if the node is the bias node
                    n.value = 1
                    # print("bias node:", n.node)
                else:
                    n.value = inputs[n.node]  # Sets the input nodes to their corresponding input

    def feedforward_calculate(self):
        for l in range(len(self.shape) + 1):
            # print("L",l)
            for n in self.nodes:
                if n.layer == l:
                    # print(n.layer,"Activated")
                    n.value = n.activation_function(n.value)
                if n.type == "bias":
                    n.value = 1
            for v in self.w:
                if v.pnode.layer == l:
                    v.tnode.value += v.pnode.value * v.value

    def collect_output_layer(self):
        outputs = []
        for n in self.nodes:
            if n.layer == len(self.shape) - 1:  # If it is on the output layer
                outputs.append(n.value)
        return outputs

    def list_internal_values(self):
        for n in self.nodes:
            print("Layer: ", n.layer, "| Node: ", n.node, "| Value: ", n.value)
        for we in self.w:
            print("PNode:", (we.pnode.layer, we.pnode.node), "| TNode:", (we.tnode.layer, we.tnode.node),
                  "| Value:", we.value)

    def calico(self, inputs, show_internals=False):  # Using an input and its weights the network returns an output

        self.set_input_nodes_to_input_values(inputs)

        self.feedforward_calculate()
        outputs = self.collect_output_layer()

        if show_internals: self.list_internal_values()

        return outputs

    def calico_from_hiddens(self, layer, values, show_internals=False):  # Starts the feedforward at a different layer
        for n in self.nodes:
            n.value = 0

            if n.layer == layer:  # If it is on hidden layer that is being manipulated
                n.value = values[n.node]  # Sets the input nodes to their corresponding input

            # print("nodevalue:",n.value)
        for l in range(len(self.shape) + 1):
            if l >= layer:
                print("L", l)
                for n in self.nodes:
                    if n.layer == l:
                        # print(n.layer,"Activated")
                        n.value = n.activation_function(n.value)
                for v in self.w:
                    if v.pnode.layer == l:
                        v.tnode.value += v.pnode.value * v.value

        outputs = []
        for n in self.nodes:
            if n.layer == len(self.shape) - 1:  # If it is on the output layer
                outputs.append(n.value)
            # if n.value == 0:
            #     print("ZERO NODE:", n.layer, n.node, n.value)

        if show_internals:
            for n in self.nodes:
                print("Layer: ", n.layer, "| Node: ", n.node, "| Value: ", n.value)
            for we in self.w:
                print("PNode:", (we.pnode.layer, we.pnode.node), "| TNode:", (we.tnode.layer, we.tnode.node),
                      "| Value:", we.value)
        return outputs

    class Node:
        def __init__(self, type, location, layer, node, use_sigmoid, draw_size):
            self.type = type
            self.color = colors[self.type]
            self.location = location
            self.layer = layer
            self.node = node
            self.value = 0
            self.draw_size = draw_size

            if self.type == "hidden":
                if not use_sigmoid:

                    self.activation_function = donata
                    self.color = (255, 128, 0)
                    # print("NOT USING SIGMOID, layer is", self.layer)
                else:

                    self.activation_function = sigmoid
                    self.color = (12, 122, 67)
                # elif self.node % 2 == 1:
                #     self.activation_function = donata
                #     self.color = (255, 128, 0)
                # elif self.node % 3 == 2:
                #     self.activation_function = square
                #     self.color = (255,255,0)
            else:
                self.activation_function = donata

        def draw(self, display):

            gfxdraw.aacircle(display, self.location[0], self.location[1], self.draw_size + 1, white)
            gfxdraw.filled_circle(display, self.location[0], self.location[1], self.draw_size + 1, white)
            gfxdraw.aacircle(display, self.location[0], self.location[1], self.draw_size, self.color)
            gfxdraw.filled_circle(display, self.location[0], self.location[1], self.draw_size, self.color)
            if self.type == "bias":
                pygame.draw.rect(display, white,
                                 [self.location[0] - self.draw_size - 1, self.location[1] - self.draw_size - 1,
                                  self.draw_size, self.draw_size * 2 + 3])
                pygame.draw.rect(display, self.color,
                                 [self.location[0] - self.draw_size, self.location[1] - self.draw_size, self.draw_size,
                                  self.draw_size * 2 + 1])

    class Edge:  # The connection between nodes with weights
        def __init__(self, value, pnode, tnode):  # Each weight has a value and connects the pnode to the tnode
            self.value = value
            self.pnode = pnode
            self.tnode = tnode  # The tnode must be one layer ahead of the pnode

        def draw(self, width, display, node_radius):
            if width > 0:
                if self.value > 0:
                    c = (0, 100, 0)
                elif self.value < 0:
                    c = (100, 0, 0)
                else:
                    c = (255, 255, 255)
                start_loc = (self.pnode.location[0] + node_radius - width, self.pnode.location[1])
                end_loc = (self.tnode.location[0] - node_radius + width, self.tnode.location[1])
                # yf.draw_line_as_polygon(display, (self.pnode.location[0] + node_radius, self.pnode.location[1]), (self.tnode.location[0] - node_radius, self.tnode.location[1]), width, (100, 100, 100))
                pgt.draw_line_as_polygon(display, start_loc, end_loc, width, c)
            # pygame.draw.line(display, black, self.pnode.location, self.tnode.location, width + 2)
            # pygame.draw.line(display, c, self.pnode.location, self.tnode.location, width)

    def show(self):
        a = []
        for v in self.w:
            a.append(v.value)
        return a

    def draw(self, size=500, show_internals=False):
        surface = pygame.Surface((500, 500))
        surface.fill(black)
        largest_weight = max([abs(v.value) for v in self.w])
        # print("Largest weight:", largest_weight, "list of weights:", self.show())

        for p in self.w:
            p.draw(round((abs(p.value) / largest_weight) * self.node_draw_size * .3, 0), surface, self.node_draw_size)
            if show_internals:
                pgt.text(surface, ((p.tnode.location[0] + p.pnode.location[0] * 1.5)/2.5,
                                   (p.tnode.location[1] + p.pnode.location[1] * 1.5)/2.5), str(round(p.value, 2)), white, 15)
        for n in self.nodes:
            n.draw(surface)
            if show_internals:
                pgt.text(surface, n.location, str(round(n.value, 2)), white, 15)
        if size != 500:
            surface = pygame.transform.scale(surface, (size, size))
        return surface
