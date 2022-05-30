import pygene
import pygame

gameDisplay = pygame.display.set_mode((500, 500))

ins = [[3],[5],[2]]
outs = [[7],[11],[5]]  # output is 2x+1 and biased nodes have to be used

guide = pygene.Species([1,1], 1, train_inputs=ins, train_outputs=outs, use_sigmoid=False, popsize=32, add_bias_nodes=True)
for _ in range(100):  # if you break up the training to one at a time you can see the network change over time.
    guide.train(1)
    gameDisplay.blit(guide.get_best_agent().draw(gameDisplay), (0, 0))
    pygame.display.update()

while True:
    testins = []
    for needed_input in range(len(ins[0])):
        testins.append(float(input("Input" + str(needed_input) +":\n")))
    print(guide.get_best_agent().calico(testins, show_internals=False))