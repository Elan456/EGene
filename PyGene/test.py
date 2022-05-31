import pygene
import pygame

gameDisplay = pygame.display.set_mode((500, 500))

testset = "x3"

ins = {"xor": [[0, 0], [0, 1], [1, 0], [1, 1]],
       "x3": [[2],[4],[3],[1]]}[testset]
outs = {"xor": [[0], [1], [1], [0]],
        "x3": [[6],[12],[9],[3]]}[testset]

guide = pygene.Species([1,2,1], 1, train_inputs=ins, train_outputs=outs, use_sigmoid=True, popsize=1000, add_bias_nodes=True)
for _ in range(100):  # if you break up the training to one at a time you can see the network change over time.
    guide.train(1)
    gameDisplay.fill((0,0,0))
    gameDisplay.blit(guide.get_best_agent().draw(size=250), (0, 0))
    pygame.display.update()

while True:
    testins = [float(a) for a in input(str(len(ins[0]))+" inputs seperated by commas:\n").split(",")]
    print(guide.get_best_agent().calico(testins, show_internals=False))
