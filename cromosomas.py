
import numpy as np
import matplotlib.pyplot as plt
import cv2



# Función de mejora de contraste basada en algoritmos genéticos
def contrast_enhancement_genetic(image):
    # Obtener el tamaño de la imagen
    height, width = image.shape[:2]

    # Definir la estructura cromosómica y los parámetros del algoritmo genético
    chromosome_size = 150 # Tamaño del cromosoma basado en el número de niveles de gris
    population_size = 300 # Tamaño de la población
    generations = 25 # Número de generaciones
    mutation_rate = 0.2  # Tasa de mutación

    # Inicializar la población de cromosomas de forma aleatoria
    population = np.random.randint(0, 256, size=(population_size, chromosome_size))

    # Función de evaluación de aptitud
    def evaluate_fitness(chromosome, image):
        # Implementar la transformación de la imagen basada en el cromosoma y calcular la aptitud
        transformed_image = np.zeros_like(image)
        for i in range(chromosome_size):
            transformed_image[image == i] = chromosome[i]
        
        # Calcular la aptitud (por ejemplo, basada en la suma de intensidades de bordes)
        fitness = np.sum(np.abs(np.gradient(transformed_image)))
        return fitness

    # Algoritmo genético: selección, cruce y mutación
    for generation in range(generations):
        # Evaluación de la aptitud de cada cromosoma en la población
        fitness_scores = [evaluate_fitness(chromosome, image) for chromosome in population]

        # Selección de los mejores cromosomas
        selected_indices = np.argsort(fitness_scores)[:population_size // 2] #np.argsort ordena de menor a mayor 
        selected_population = population[selected_indices]# se elige de el primer grupo que se acercan a 0 = negro

        # Cruce (por ejemplo, un punto de cruce)
        crossover_point = chromosome_size // 2
        offspring = np.empty_like(population)
        for i in range(population_size):
            parent_indices = np.random.choice(len(selected_population), 2, replace=False)
            parent1, parent2 = selected_population[parent_indices] 
            offspring[i] = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

        # Mutación
        mutation_mask = np.random.rand(*offspring.shape) < mutation_rate
        random_mutations = np.random.randint(0, 256, size=offspring.shape)
        offspring = np.where(mutation_mask, random_mutations, offspring)

        # Reemplazar la población anterior con la nueva generación
        population = offspring

    # Seleccionar el mejor cromosoma de la última generación
    best_chromosome = population[np.argmin([evaluate_fitness(chromosome, image) for chromosome in population])]

    # Aplicar el mejor cromosoma a la imagen original
    enhanced_image = np.zeros_like(image)
    for i in range(chromosome_size):
        enhanced_image[image == i] = best_chromosome[i]

    return enhanced_image


# Cargar la imagen
ruta_imagen = r"C:\Users\piero\Downloads\imagenes\mono.jpg"

# Cargar la imagen en escala de grises desde la ruta especificada
image = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

# Aplicar el método de mejora de contraste basado en algoritmos genéticos
enhanced_image = contrast_enhancement_genetic(image)

# Mostrar las imágenes original y mejorada
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(enhanced_image, cmap='gray')
plt.title('Imagen Mejorada')
plt.axis('off')

plt.show()
