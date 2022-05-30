import pygene
import pygame

ins = [[3], [5], [1]]
outs = [[6], [10], [2]]

guide = pygene.Species([1, 2, 3,3,1], .1, train_inputs=ins, train_outputs=outs)
