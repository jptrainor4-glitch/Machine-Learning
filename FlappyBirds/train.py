import neat 
from fb import main, load_images, Bird, WIN_HEIGHT, pygame
import matplotlib.pyplot as plt

gen = 0
fitness_history_best = []
fitness_history_avg = []
hall_of_fame = []
best_genome = 0
show_network = False
show_graph = False

def eval_genomes(genomes, config):
    global gen, best_genome, fitness_history_best, fitness_history_avg, show_network, show_graph
    gen += 1
    networks = []
    genomes_list = []
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        networks.append(net)
        genomes_list.append(genome)

    if gen < 15:
        config.genome_config.weight_mutate_power = 0.9
        config.genome_config.conn_add_prob   = 0.9
        config.genome_config.conn_delete_prob   = 0.1
        config.genome_config.bias_mutate_rate = 0.7
        config.genome_config.node_add_prob = 0.2
        config.reproduction_config.elitism  = 5
        config.reproduction_config.survival_threshold = 0.3
    elif gen < 20:
        config.genome_config.weight_mutate_power = 0.7
        config.genome_config.bias_mutate_rate        = 0.3
        config.genome_config.conn_delete_prob   = 0.2
        config.genome_config.node_add_prob = 0.5
        config.reproduction_config.elitism  = 3
        config.reproduction_config.survival_threshold = 0.4
    elif gen < 30:
        config.genome_config.weight_mutate_power = 0.6
        config.genome_config.node_add_prob = 0.2
        config.genome_config.node_delete_prob = 0.2
        config.genome_config.conn_add_prob   = 0.5
        config.genome_config.conn_delete_prob   = 0.2
        config.reproduction_config.survival_threshold = 0.3
    elif gen < 40:
        config.genome_config.weight_mutate_power = 0.3
        config.genome_config.conn_add_prob   = 0.2
    else:
        config.reproduction_config.survival_threshold = 0.2

    if gen > 1:
        best_genome = max(genomes, key=lambda x: x[1].fitness)[1]

    show_network, show_graph = main(networks, genomes_list, gen, best_genome, config, show_network, show_graph, fitness_history_best, fitness_history_avg)

    fitnesses = [g.fitness for _, g in genomes]
    fitness_history_best.append(max(fitnesses))
    fitness_history_avg.append(sum(fitnesses) / len(fitnesses))

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config.txt')

population = neat.Population(config)

#population.add_reporter(neat.StdOutReporter(True))
#population.add_reporter(neat.StatisticsReporter())

population.run(eval_genomes, 100)  # run for 50 generations
pygame.quit()

plt.plot(fitness_history_best, label='Best')
plt.plot(fitness_history_avg, label='Average')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.show()

#export generation number varable to fb.py and display on screen
#Save average fitness and best fitness in a dictionary with generation number and display a graph at end of program