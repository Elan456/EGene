import copy
import random
import math

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
          "output": (0, 150, 0)}


def draw_line_as_polygon(gameDisplay, startpos, endpos, width,
                         color, aa=True):  # Wide lines look ugly compared to polygons this draws a polygon as a line
    startx, starty = startpos
    endx, endy = endpos
    angle = math.atan2(endy - starty, endx - startx)
    perpangle = angle - math.pi / 2

    m = math

    coords = [(startx + math.cos(perpangle) * width, starty + m.sin(perpangle) * width),
              (startx + math.cos(perpangle) * -1 * width, starty + m.sin(perpangle) * -1 * width),
              (endx + math.cos(perpangle) * -1 * width, endy + m.sin(perpangle) * -1 * width),
              (endx + math.cos(perpangle) * width, endy + m.sin(perpangle) * width)]

    pygame.draw.polygon(gameDisplay, color, coords)
    if aa:
        gfxdraw.aapolygon(gameDisplay, coords, color)


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
    scorefunction, agent = t
    return scorefunction(agent)


class Species:
    def __init__(self, changerate, popsize=32, train_inputs=None, train_outputs=None, scorefunction=None,
                 initalweights=None, draw_window=False, layer=1, datapergen=None, shape=None, metapop=0,
                 use_sigmoid=True, can_change_changerate=True, use_multiprocessing=True, set_all_zero=False):
        self.use_multiprocessing = use_multiprocessing
        self.use_sigmoid = use_sigmoid
        self.can_change_changerate = can_change_changerate
        self.set_all_zero = set_all_zero

        self.epochs = 0  # Count of all epochs every trained on this species
        self.lowestlost = float("inf")
        self.all_blost = []  # All the lowest losses for each epoch

        self.metapop = metapop
        if scorefunction == None and train_inputs == None:
            raise CustomError("Species needs either a list of inputs + outputs or a scorefunction")
        if scorefunction == None:
            self.n_inputs = len(train_inputs[0])
            self.n_outputs = len(train_outputs[0])
            self.scorefunction = self.evaluate
            self.using_custom_score_function = False

        else:
            self.scorefunction = scorefunction
            self.using_custom_score_function = True

        if shape == None:
            self.shape = [self.n_inputs + 1,  # plus 1 for bias
                          self.n_inputs * 2,

                          self.n_outputs]
        else:
            self.shape = shape
        if not self.using_custom_score_function:
            if self.shape[0] != self.n_inputs + 1:
                raise CustomError("First layer node count does not equal inputs number + 1 (the plus 1 is for bias)")
            if self.shape[-1] != self.n_outputs:
                raise CustomError("Last layer node count does not equal outputs")
            if duplicatechecker(train_inputs):
                print("--Duplicate Inputs Found--")
            if datapergen == None:
                datapergen = len(train_inputs)
            self.datapergen = min(len(train_inputs), datapergen)
        else:
            # print("Datepergen set to None because self.using_custom... is", self.using_custom_score_function)
            self.datapergen = None

        self.layer = layer
        self.draw_window = draw_window
        self.initalweights = initalweights

        # print("datagennum:", self.datapergen)

        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.popsize = popsize
        self.changerate = changerate

        self.agents = []

        for p in range(popsize):  # Creating first generation of agents
            self.agents.append(Agent(self.shape, draw_window, self.use_sigmoid, self.set_all_zero))
        print("--First agents made")
        # INITAL WEIGHTS
        if self.initalweights is not None and self.set_all_zero is False:
            for v in range(len(self.agents[0].w)):
                self.agents[0].w[v].value = self.initalweights[v]
        # print("-- Initial Weights initalized")
        # for p in self.agents:
        #     print(p.calico([2, 3]))

    def evaluate(self, p, inputs,
                 output):  # Using an input it calculates the loss or score of the agent default scorefuction
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

    def supereval(self, agent):  # Will test the network after 10 generations of changes
        if self.using_custom_score_function:
            sent_score_function = self.scorefunction
        else:
            sent_score_function = None
        metaspecies = Species(.1, layer=self.layer + 1, popsize=self.metapop, train_inputs=self.train_inputs,
                              train_outputs=self.train_outputs,
                              initalweights=agent.show(), datapergen=self.datapergen, shape=self.shape,
                              scorefunction=sent_score_function,
                              use_sigmoid=self.use_sigmoid)  # extra layer to keep track of recursion depth
        # print("\n\nEntering metalayer\n\n")
        metaspecies.train(10, 10)
        # print("\n\nExiting metalayer\n\n")
        # print("premeta:")
        # self.evaluate(train_inputs, train_outputs)
        # print(self.loss, self.w)
        # print("metabest:")
        # print(metaspecies.agents[0].loss, metaspecies.agents[0].w)
        # print()
        for v in range(len(agent.w)):
            agent.w[v].value = metaspecies.agents[0].w[v].value

        # print("\nMETAW:")
        metaspecies.agents[0].show()
        agent.loss = metaspecies.agents[0].loss
        return agent

    def scoreall(self, show, scorefunction):  # Evaluates all of the agents and puts them in order from best to worst
        # print("SCORE")
        # MUST PICK WHICH INPUTS AND OUTPUTS WILL BE USED FOR EVALUATION
        if not self.using_custom_score_function:  # Chooses the input output stuff
            inout = []  # List of tuples with each tuple having input, output
            for i in range(len(self.train_inputs)):
                inout.append((self.train_inputs[i], self.train_outputs[i]))
            random.shuffle(inout)
            ins = []
            outs = []
            for i in range(self.datapergen):
                ins.append(inout[i][0])
                outs.append(inout[i][1])

        if self.using_custom_score_function:  # if a custom function is being use
            # print("MANAGER ABOUT TO BE MADE")
            # manager = multiprocessing.Manager()
            # return_dict = manager.dict()
            # sections = breaklist(self.agents, 16)
            # print("SECTIONS CREATED\n")

            if self.use_multiprocessing:
                jobs = []
                # print("JOBS LIST MADE")
                j = 0
                data = []
                p = Pool()
                for a in self.agents:
                    data.append((self.scorefunction, a))

                results = p.map(custom_eval, data)

                for a in range(len(self.agents)):
                    self.agents[a].loss = results[a]
                print("---")
                print("AVG loss:", sum(results) / len(results), "in order:", sorted(results))
            else:
                for a in range(len(self.agents)):
                    self.agents[a].loss = self.scorefunction(self.agents[a])

        else:
            # print("ins", ins, "outs:", outs)

            for p in self.agents:
                p.loss = scorefunction(p, ins, outs)

                if self.layer <= 1 and self.metapop != 0:
                    p = self.supereval(p)

        self.agents.sort(key=lambda x: x.loss)
        # for p in self.agents:
        #     print(p.loss, p.show())
        # if show and self.layer == 1:
        #     print(self.agents[0].loss, self.agents[0].show())

        # time.sleep(.2)

    def crossover(self, p1, p2):  # Crosses the weights of two parents to get a new child
        child = Agent(self.shape, self.draw_window, self.use_sigmoid)
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

    def mutate(self, agent):
        # print("Before mutation:", [w.value for w in agent.w])
        for w in agent.w:
            w.value += random.random() * random.choice([-1, 0, 0, 0, 1]) * self.changerate
        # print("After  mutation:", [w.value for w in agent.w])
        return agent

    def nextgen(self):  # Gives all of the agents a list of weights similar to the best weight
        choices = []
        # print("NEWGEN")
        # print("\n\n")

        # print(choices)
        best_weights = self.agents[0].w
        n = 0
        # print()
        # print("ENTIRE POPULATION BEFORE CHANGES")
        # for p in self.agents:
        #     print([w.value for w in p.w])
        max_index = max(4, int(self.popsize / 16))  # The worst agent possible for reproduction

        for p in range(len(self.agents) // 16, len(self.agents)):  # Only the worst 15/16 are changed

            n += 1
            while True:  # Choosing the two parents
                # print("0 to", max(2, int(self.popsize / 16)))

                p1 = self.agents[random.randint(0, max_index)]
                p2 = self.agents[random.randint(0, max_index)]
                # print(p1,p2)

                if p1 != p2:
                    break
            # print("p1,p2", p1, p2)

            self.agents[p] = self.crossover(p1, p2)  # Crosses the parents to produce a child
            self.agents[p] = self.mutate(self.agents[p])  # Mutates the child based on the change rate
        # print("ENTIRE POPULATION AFTER CHANGES")
        # for p in self.agents:
        #     print([w.value for w in p.w])

        # print("Changed:",n)
        # print("Newval:", p.w[j].value)
        # print(int(len(self.agents)*(15/16)))

        for p in range(int(len(self.agents) * (15 / 16)), len(self.agents)):  # Last 16th are just asexual mutants of the best 16th only one weight is changed
            p1 = self.agents[random.randint(0, max_index)]
            w_index = random.randint(0, len(self.agents[0].w) - 1)
            # for i, v in enumerate(self.agents[p].w):
            self.agents[p].w = copy.deepcopy(p1.w)
            self.agents[p].w[w_index].value = random.choice([-1, 1]) * random.random() * self.changerate + p1.w[w_index].value
            # print("p1 stuff:", [w.value for w in p1.w])
            # print("ne stuff:", [w.value for w in self.agents[p].w])
        # print("randomized:",n)

        # print("bestweights", best_weights)
        # for p in self.agents:
        #     # print("old", p.show())
        #     if p.w != best_weights:
        #         new_weights = []
        #         choice = random.choice(choices)
        #         if choice == "randomnew":
        #             #print("rand")
        #             for v in p.w:
        #                 v.value = random.choice([-1, 1]) * round(random.random(), 2)
        #             # print(new_weights)
        #         elif choice == "smallchange":
        #             # print("small")
        #             for v in range(len(p.w)):
        #                 # print(v)
        #                 p.w[v].value = best_weights[v].value + (
        #                             random.random() * random.choice([-1, 1]) * self.changerate)
        # else:
        #     print("BEST")
        # print("new", p.show())

    def train(self, epochs, show_pop=False):

        for v in range(epochs):

            # print("\n\n Epoch:",v,"\n\n")

            self.scoreall(True, self.scorefunction)

            if self.draw_window and self.layer == 1:
                self.agents[0].draw(gameDisplay)

                # pygame.display.update()

            blost = self.agents[0].loss  # blost is the lost of the best agent in this epoch
            if self.layer == 1:
                # if blost <= self.lowestlost:  # lowestlost is the lowestlost ever
                #     if len(self.all_blost) > 6 and abs(
                #             blost - all_blost[v - 5]) < self.changerate / 100 and self.can_change_changerate:
                #         pass
                #         # print("CONVERGED", "changerate:", self.changerate)
                #         # break
                #     lowestlost = blost
                self.all_blost.append(self.agents[0].loss)
                print(self.epochs, ":", "loss:", self.agents[0].loss, self.agents[0].show())
                # print("-2:", all_blost[v-2], "v:", all_blost[v])
                if len(self.all_blost) > 2 and self.all_blost[self.epochs - 2] == self.all_blost[self.epochs] and self.can_change_changerate:
                    self.changerate /= 2
                    print("--\nNo change from 2 gens ago so changerate is being lowered to", self.changerate, "\n--")

            if show_pop:
                all_losses = [a.loss for a in self.agents]
                print("Avg:", sum(all_losses) / len(all_losses), "Losses:", all_losses)
            self.epochs += 1
            self.nextgen()

    # def train(self, gencount=100):
    def get_best_agent(self):
        return self.agents[0]


class Agent:
    def __init__(self, shape, draw_window, use_sigmoid, set_all_zero=False):
        self.loss = 0
        self.use_sigmoid = use_sigmoid
        self.draw_window = draw_window
        self.set_all_zero = set_all_zero
        self.shape = shape

        # Initate nodes ------------
        self.nodes = []
        xscale = 500 / (len(self.shape) + 1)

        xstart = xscale
        layer_starts = []  # Keeps track of where each layer starts so weights are created faster

        # Determining the radius of the nodes when drawn so they all fit
        self.node_draw_size = min(int(450 / max(self.shape) / 2.2), int(xscale / 10))
        # print("creating nodes")
        for l in range(len(self.shape)):  # Each layer
            layer_starts.append(len(self.nodes))
            x = int(l * xscale + xstart)
            layersize = self.shape[l]
            for n in range(self.shape[l]):  # Each node in the layer:
                yscale = 500 / (layersize + 1)
                y = int(n * yscale + yscale)
                if l == 0:
                    type = "input"
                elif l == len(self.shape) - 1:
                    type = "output"
                else:
                    type = "hidden"
                self.nodes.append(Agent.Node(type, (x, y), l, n, self.use_sigmoid, self.node_draw_size))

        layer_starts.append(len(self.nodes))
        # print("nodes made")
        # print([n.layer for n in self.nodes])
        # print("layer_Starts:", layer_starts)

        # Initate weights
        # print("creating weigths")
        timer = time.time()
        things_checked = 0
        self.w = []
        for n in self.nodes:
            if n.layer != len(layer_starts) - 2:
                # print(layer_starts[n.layer+1],"to", layer_starts[n.layer+2])
                for target_index in range(layer_starts[n.layer + 1], layer_starts[n.layer + 2]):  # All nodes ahead in index are checked
                    things_checked += 1
                    t = self.nodes[target_index]
                    if t.layer == n.layer + 1:  # If the target node is one layer ahead of the current node
                        if self.set_all_zero is False:
                            self.w.append(Agent.Wij(random.choice([-1, 1]) * random.random(), n, t))
                        else:
                            self.w.append(Agent.Wij(0, n, t))
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
                if n.node == self.shape[0] - 1:  # Checks if the node is the bias node
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

    def calico(self, inputs, show_internals=False):  # Using an input and its weights the agent returns an output

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

    class Wij:  # The connection between nodes with weights
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
                draw_line_as_polygon(display, start_loc, end_loc, width, c)
            # pygame.draw.line(display, black, self.pnode.location, self.tnode.location, width + 2)
            # pygame.draw.line(display, c, self.pnode.location, self.tnode.location, width)

    def show(self):
        a = []
        for v in self.w:
            a.append(v.value)
        return a

    def draw(self, surface):
        surface.fill(black)
        largest_weight = max([abs(v.value) for v in self.w])
        # print("Largest weight:", largest_weight, "list of weights:", self.show())

        for p in self.w:
            p.draw(round((abs(p.value) / largest_weight) * self.node_draw_size * .3, 0), surface, self.node_draw_size)
        for n in self.nodes:
            n.draw(surface)
        return surface
