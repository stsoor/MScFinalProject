import numpy as np
from problem.model import ProblemModel
from solution.drawing import HypergraphDrawer

class GA: # genetic Algorithm (instead of fitness value we use an objective function [lower value is better] here too)
    def __init__(self, lower_bounds, upper_bounds, initializer, evaluator, population_size, selection_pct, mutation_pct, mutation_random_generator, max_iteration_num, crossover_pct=0.5, debug=None):
        #assert int(selection_pct * population_size) * int(1.0 / selection_pct) == population_size, 'the given percentage would yield inconsistent population size'
        #pass
        assert selection_pct <= 0.5
        assert int(population_size - population_size*selection_pct) % 2 == 0, 'the number of not selected population has to be even'
        assert len(lower_bounds) == len(upper_bounds)

    def _selection(self):
        raise NotImplementedError

    def _crossover(self, selected_indices):
        raise NotImplementedError

    def _mutation(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError

class NaiveGA(GA):
    def __init__(self, lower_bounds, upper_bounds, initializer, evaluator, population_size, selection_pct, mutation_pct, mutation_random_generator, max_iteration_num, crossover_pct=0.5, debug=None):
        super().__init__(lower_bounds, upper_bounds, initializer, evaluator, population_size, selection_pct, mutation_pct, mutation_random_generator, max_iteration_num, crossover_pct, debug)
        self.evaluator = evaluator
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.initializer = initializer
        self.selection_pct = np.float32(selection_pct)
        self.mutation_pct = np.float32(mutation_pct)
        self.mutation_random_generator = mutation_random_generator
        self.crossover_pct = crossover_pct
        self.max_iteration_num = max_iteration_num

        self.debug = debug
        
        self.population_size = population_size
        self.dimension = len(lower_bounds)

    def _setup_optimization(self):
        self.x = self.initializer(self.population_size) # current particle positions
        self.fitness_values = np.zeros(self.population_size, dtype=np.float32)
        self.best_position = np.array([], dtype=self.x.dtype)
        self.best_value = np.inf

        self._update()

    def _selection(self):
        num_selected = int(self.selection_pct * self.population_size)
        worst_indices = self.fitness_values.argsort()[-num_selected:]
        self.x[worst_indices, :] = self.initializer(len(worst_indices))
        self._update(worst_indices)
        selected_indices = self.fitness_values.argsort()[:num_selected]
        return selected_indices

    def _select_parent_ids(self, parent_pair_num):
        def add_second_parent(parent_1_array, fitness_implied_probabilities, fitness_values_sum):
            parent_1_id = parent_1_array[0]
            parent_1_fitness = fitness_implied_probabilities[parent_1_id] * fitness_values_sum
            probabilities_without_parent_1 = fitness_implied_probabilities * fitness_values_sum / (fitness_values_sum - parent_1_fitness)
            probabilities_without_parent_1[parent_1_id] = 0.0
            new_row = np.array([parent_1_id, np.random.choice(np.arange(self.population_size), p=probabilities_without_parent_1)], dtype=np.int32)
            return new_row

        ascending_fitness_values = (self.fitness_values.max() - self.fitness_values)
        fitness_values_sum = ascending_fitness_values.sum()
        fitness_implied_probabilities = ascending_fitness_values / fitness_values_sum
        parent_1s = np.random.choice(np.arange(self.population_size), size=(parent_pair_num,1), p=fitness_implied_probabilities, replace=True)
        parents = np.apply_along_axis(add_second_parent, 1, parent_1s, fitness_implied_probabilities, fitness_values_sum)
        return parents
    
    def _permute_gene_ids(self, parent_pair_num, parent_ids):
        id_grid = np.tile(np.arange(self.dimension),(parent_pair_num, 1))
        np.apply_along_axis(np.random.shuffle, 1, id_grid) # does it in place, so the columns of id_grid are shuffled now (row by row)
        return id_grid

    def _combine_parents(self, parent_1_id, parent_2_id, gene_id_permutation):
        parent_2_gene_ids = gene_id_permutation[-int(self.dimension*self.crossover_pct):]
        new_row = self.x[parent_1_id, :].copy()
        new_row[parent_2_gene_ids] = self.x[parent_2_id, parent_2_gene_ids]
        return new_row

    def _crossover(self, selected_indices):
        first_children_wrapper = lambda combination_table_row : self._combine_parents(combination_table_row[0], combination_table_row[1], combination_table_row[2:])
        second_children_wrapper = lambda combination_table_row : self._combine_parents(combination_table_row[1], combination_table_row[0], combination_table_row[2:])

        children_num = self.population_size - len(selected_indices)
        #indices = np.arange(children_num).reshape(children_num, 1)
        parent_ids = self._select_parent_ids(children_num//2)
        combination_table = np.hstack((parent_ids, self._permute_gene_ids(children_num//2, parent_ids)))
        first_children = np.apply_along_axis(first_children_wrapper, 1, combination_table)
        second_children = np.apply_along_axis(second_children_wrapper, 1, combination_table)
        best_values = self.x[selected_indices, :].copy()

        new_population = np.vstack((best_values, first_children, second_children))

        return new_population

    def _mutation(self, population, selected_indices):
        mutation = self.mutation_random_generator(size=population.shape)
        mutation[selected_indices, :] = 0
        mutation[np.random.random(size=mutation.shape) > self.mutation_pct] = 0
        population += mutation
        np.clip(population, self.lower_bounds, self.upper_bounds, out=population)
        return population

    def _update_fitness_values(self, x_row_indices=None):
        x_row_indices = np.arange(self.x.shape[0]) if x_row_indices is None else x_row_indices
        evaluation = np.apply_along_axis(self.evaluator, 1, self.x[x_row_indices, :]).astype(np.float32)
        self.fitness_values[x_row_indices] = evaluation[:,0] if len(evaluation.shape) > 1 else evaluation
        return evaluation

    def _update_bests(self):
        argmin = np.argmin(self.fitness_values)
        if self.fitness_values[argmin] < self.best_value:
            self.best_value = self.fitness_values[argmin]
            self.best_position = self.x[argmin, :].copy()

            return True
        return False

    def _update(self, x_row_indices=None):
        self._update_fitness_values()
        is_best_updated = self._update_bests()
        if self.debug is not None and is_best_updated:
            print(self.best_value)

    def _run(self):
        iteration = 0
        while iteration < self.max_iteration_num:
            iteration += 1
            selected_indices = self._selection()
            new_population = self._crossover(selected_indices)
            self.x = self._mutation(new_population, selected_indices)
            self._update()

            if self.debug is not None:
                print(iteration)

        return self.best_value, self.best_position, iteration

    def __call__(self):
        self._setup_optimization()
        return self._run()

class NaiveMultiRowGA(NaiveGA):
    def __init__(self, row_size, lower_bounds, upper_bounds, initializer, evaluator, population_size, selection_pct, mutation_pct, mutation_random_generator, max_iteration_num, crossover_pct=0.5, debug=None):
        assert len(lower_bounds) >= row_size
        assert len(lower_bounds) % row_size == 0
        super().__init__(lower_bounds, upper_bounds, initializer, evaluator, population_size, selection_pct, mutation_pct, mutation_random_generator, max_iteration_num, crossover_pct, debug)
        self.row_size = row_size

    def _permute_gene_ids(self, parent_pair_num, parent_ids): # override
        id_grid = np.tile(np.arange(self.row_size),(parent_pair_num, 1)) # assertion is in place so children is always even
        np.apply_along_axis(np.random.shuffle, 1, id_grid) # does it in place, so the columns of id_grid are shuffled now (row by row)
        return id_grid

    def _combine_parents(self, parent_1_id, parent_2_id, gene_id_permutation): # override
        parent_2_row_ids = gene_id_permutation[-int(self.row_size*self.crossover_pct):]
        row_num = int(self.dimension // self.row_size)
        starting_points = tuple([parent_2_row_ids + i*self.row_size for i in range(row_num)])
        parent_2_gene_ids = np.hstack(starting_points)
        new_row = self.x[parent_1_id, :].copy()
        new_row[parent_2_gene_ids] = self.x[parent_2_id, parent_2_gene_ids]
        return new_row

class NaiveMultiRowHypergraphGA(NaiveMultiRowGA):
    def __init__(self, row_size, lower_bounds, upper_bounds, initializer, evaluator, population_size, selection_pct, mutation_pct, mutation_random_generator, max_iteration_num, crossover_pct=0.5, debug=None, problem=None):
        super().__init__(row_size, lower_bounds, upper_bounds, initializer, evaluator, population_size, selection_pct, mutation_pct, mutation_random_generator, max_iteration_num, crossover_pct, debug)
        self.problem = problem

    def _setup_optimization(self): # override
        edge_num = self.problem.hypergraph.shape[1]
        self.edgewise_fitness_values = np.zeros((self.population_size, edge_num), dtype=np.float32)
        super()._setup_optimization()

    def _update_fitness_values(self):
        evaluation = super()._update_fitness_values()
        self.edgewise_fitness_values = evaluation[:,1:]

    def _show_debug_info(self):
        if self.debug is not None and self.problem is not None:
            drawer = HypergraphDrawer(self.problem, self.best_position)
            drawer.show(wait_key=self.debug)
            print(self.best_value)

    def _update_bests(self):
        is_updated = super()._update_bests()
        if is_updated:
            self._show_debug_info()

class EdgewiseHypergraphGA(NaiveMultiRowHypergraphGA): # TODO crossover pct for gene selection (currently always 0.5)
    def __init__(self, initializer, evaluator, population_size, selection_pct, mutation_pct, mutation_random_generator, max_iteration_num, debug=None, problem=None):
        assert isinstance(problem, ProblemModel)
        super().__init__(problem.hypergraph.shape[0], problem.get_vector_lower_bounds(), problem.get_vector_upper_bounds(), initializer, evaluator, population_size, selection_pct, mutation_pct, mutation_random_generator, max_iteration_num, crossover_pct, debug, problem)

    def _select_gene_parents(self, parent_ids):
        def ensure_equal_inheritance(parent_1_gene_mask, parent_nodewise_fitness_values_difference):
            parent_1_gene_num = parent_1_gene_mask.sum()
            if parent_1_gene_num > parent_1_gene_mask.size // 2:
                to_flip_num = parent_1_gene_num - (parent_1_gene_mask.size // 2)
                argsort = np.argsort(parent_nodewise_fitness_values_difference[parent_1_gene_mask])
                to_flip_args = argsort[:to_flip_num]
                to_flip_genes = np.where(parent_1_gene_mask)[0][to_flip_args]
                parent_1_gene_mask[to_flip_genes] = False
            else:
                parent_2_gene_mask = ~parent_1_gene_mask
                to_flip_num = (parent_1_gene_mask.size // 2) - parent_1_gene_num
                argsort = np.argsort(parent_nodewise_fitness_values_difference[parent_2_gene_mask])
                to_flip_args = argsort[:to_flip_num]
                to_flip_genes = np.where(parent_2_gene_mask)[0][to_flip_args]
                parent_2_gene_mask[to_flip_genes] = False
                parent_1_gene_mask = ~parent_2_gene_mask
            return parent_1_gene_mask

        parent_1_edgewise_fitness_values = np.apply_along_axis(lambda parent_pair: self.edgewise_fitness_values[parent_pair[0], :], 1, parent_ids)
        parent_2_edgewise_fitness_values = np.apply_along_axis(lambda parent_pair: self.edgewise_fitness_values[parent_pair[1], :], 1, parent_ids)
        edge_counts = np.apply_along_axis(np.sum, 1, self.problem.hypergraph)
        edge_counts[edge_counts == 0] = 1
        parent_1_nodewise_fitness_values = np.apply_along_axis(lambda edgewise_fitness, edge_counts: (self.problem.hypergraph @ edgewise_fitness) / edge_counts, 1, parent_1_edgewise_fitness_values, edge_counts)
        parent_2_nodewise_fitness_values = np.apply_along_axis(lambda edgewise_fitness, edge_counts: (self.problem.hypergraph @ edgewise_fitness) / edge_counts, 1, parent_2_edgewise_fitness_values, edge_counts)
        parent_nodewise_fitness_values_differences = np.abs(parent_1_nodewise_fitness_values - parent_2_nodewise_fitness_values)

        parent_1_gene_masks = (parent_1_nodewise_fitness_values <= parent_2_nodewise_fitness_values)
        for row in range(len(parent_1_gene_masks)):
            parent_1_gene_masks[row,:] = ensure_equal_inheritance(parent_1_gene_masks[row], parent_nodewise_fitness_values_differences[row])
        
        return parent_1_gene_masks

    def _combine_parents(self, parent_1_id, parent_2_id, parent_1_gene_mask):
        full_mask = np.hstack((parent_1_gene_mask, parent_1_gene_mask, parent_1_gene_mask))
        new_row = self.x[parent_2_id, :].copy()
        new_row[full_mask] = self.x[parent_1_id, full_mask]
        return new_row

    def _crossover(self, selected_indices):
        children_num = self.population_size - len(selected_indices)
        parent_ids = self._select_parent_ids(children_num)
        parent_1_gene_masks = self._select_gene_parents(parent_ids)
        
        children = np.empty((self.x.shape[0] - len(selected_indices), self.x.shape[1]), dtype=np.float32)
        for row in range(len(parent_1_gene_masks)):
            parent_1_id, parent_2_id = parent_ids[row,:]
            children[row,:] = self._combine_parents(parent_1_id, parent_2_id, parent_1_gene_masks[row])
        best_values = self.x[selected_indices, :].copy()
        new_population = np.vstack((best_values, children))

        return new_population
