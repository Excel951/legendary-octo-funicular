import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Fungsi untuk menghasilkan populasi awal
def generate_population(n_population, n_features):
    population = np.random.randint(2, size=(n_population, n_features))
    # Pastikan setiap solusi memiliki setidaknya satu fitur terpilih
    for i in range(n_population):
        while np.sum(population[i]) == 0:
            population[i] = np.random.randint(2, size=n_features)
    return population


# Fungsi fitness (menggunakan akurasi model)
def fitness_function(solution, X_train, X_test, y_train, y_test):
    selected_features = solution == 1
    if np.sum(selected_features) == 0:
        return 0  # Jika tidak ada fitur yang dipilih, kembalikan fitness 0

    # Gunakan RandomForestClassifier sebagai model evaluasi
    model = RandomForestClassifier()
    model.fit(X_train[:, selected_features], y_train)
    predictions = model.predict(X_test[:, selected_features])
    return accuracy_score(y_test, predictions)


# Fungsi untuk fase grup (seleksi solusi terbaik dalam grup)
def group_stage(population, fitness_scores, group_size):
    selected_solutions = []
    for i in range(0, len(population), group_size):
        group = population[i:i + group_size]
        group_fitness = fitness_scores[i:i + group_size]
        if len(group) > 0:  # Pastikan grup tidak kosong
            winner_index = np.argmax(group_fitness)
            selected_solutions.append(group[winner_index])

    # Jika jumlah solusi ganjil, tambahkan solusi terbaik lagi
    if len(selected_solutions) % 2 != 0:
        selected_solutions.append(selected_solutions[np.argmax(fitness_scores)])

    return np.array(selected_solutions)


# Fungsi crossover (menggabungkan dua solusi)
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    # Pastikan anak memiliki setidaknya satu fitur terpilih
    if np.sum(child1) == 0:
        child1[np.random.randint(len(child1))] = 1
    if np.sum(child2) == 0:
        child2[np.random.randint(len(child2))] = 1
    return child1, child2


# Fungsi mutasi (mengubah beberapa gen secara acak)
def mutate(solution, mutation_rate):
    for i in range(len(solution)):
        if np.random.rand() < mutation_rate:
            solution[i] = 1 - solution[i]
    # Pastikan solusi memiliki setidaknya satu fitur terpilih
    if np.sum(solution) == 0:
        solution[np.random.randint(len(solution))] = 1
    return solution


# Fungsi fase knockout (crossover dan mutasi)
def knockout_stage(selected_solutions, mutation_rate):
    new_population = []
    for i in range(0, len(selected_solutions), 2):
        if i + 1 < len(selected_solutions):
            parent1, parent2 = selected_solutions[i], selected_solutions[i + 1]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        else:
            # Jika jumlah solusi ganjil, mutasi solusi terakhir
            child = mutate(selected_solutions[i], mutation_rate)
            new_population.append(child)
    return np.array(new_population)


# Algoritma World Cup Optimization (WCO)
def world_cup_optimization(X, y, n_population=20, n_generations=10, group_size=4, mutation_rate=0.1):
    n_features = X.shape[1]
    population = generate_population(n_population, n_features)

    # Bagi data menjadi training dan testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for generation in range(n_generations):
        # Evaluasi fitness untuk setiap solusi
        fitness_scores = np.array(
            [fitness_function(solution, X_train, X_test, y_train, y_test) for solution in population])

        # Fase grup
        selected_solutions = group_stage(population, fitness_scores, group_size)

        # Fase knockout (crossover dan mutasi)
        if len(selected_solutions) > 0:
            population = knockout_stage(selected_solutions, mutation_rate)
        else:
            # Jika tidak ada solusi yang lolos, gunakan populasi awal
            population = generate_population(n_population, n_features)

        # Cetak fitness terbaik setiap generasi
        if len(fitness_scores) > 0:
            best_fitness = np.max(fitness_scores)
            print(f"Generasi {generation + 1}, Ukuran Populasi: {len(population)}, Fitness Terbaik: {best_fitness}")
        else:
            print(f"Generasi {generation + 1}, Ukuran Populasi: {len(population)}, Fitness Scores: Kosong")

    # Pilih solusi terbaik
    if len(population) > 0:
        best_solution = population[np.argmax(fitness_scores)]
        return best_solution
    else:
        print("Tidak ada solusi yang valid.")
        return None


# Contoh penggunaan
if __name__ == "__main__":
    # Generate dataset dummy (300 record, 30 fitur)
    np.random.seed(42)
    X = np.random.rand(300, 30)
    y = np.random.randint(2, size=300)  # Binary classification

    # Jalankan WCO untuk feature selection
    best_solution = world_cup_optimization(X, y, n_population=20, n_generations=10, group_size=4, mutation_rate=0.1)

    # Tampilkan hasil
    if best_solution is not None:
        print("Solusi Terbaik (Fitur Terpilih):", best_solution)
        print("Jumlah Fitur Terpilih:", np.sum(best_solution))