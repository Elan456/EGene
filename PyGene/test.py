import pygene
import pygame

gameDisplay = pygame.display.set_mode((500, 500))

ins = [[1,2,4,8], [2,4,8,16], [3,6,12,24]]
#outs = [[1,2,3,4,5,6,7,8,9,18], [2,3,4,5,6,7,8,9,10,20]]


guide = pygene.Species([4,1,2,4], 1, train_inputs=ins, train_outputs=ins, use_sigmoid=False, popsize=32, add_bias_nodes=True)
guide.train(20)
gameDisplay.blit(guide.get_best_agent().draw(gameDisplay), (0, 0))
pygame.display.update()
print(guide.get_best_agent().calico([4,8,16,32], show_internals=False))
a = input()