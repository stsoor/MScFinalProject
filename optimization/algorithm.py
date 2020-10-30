import numpy as np

from solution.drawing import HypergraphDrawer

class PSO: # Particle Swarm Optimization
    def __init__(self, problem, initializer, evaluator, particle_num, w, c_1, c_2, max_iteration_num, min_value_change=1e-8, debug_wait_key=None):
        # velocity_(T+1) = w*velocity_(T) + c_1*random_1*(best_particle_pos-current_pos) + c_2*random_2*(best_global_pos-current_pos)
        self.evaluator = evaluator
        self.problem = problem
        self.initializer = initializer
        self.lower_bounds = problem.get_vector_lower_bounds()
        self.upper_bounds = problem.get_vector_upper_bounds()
        self.w = np.float32(w)
        self.c_1 = np.float32(c_1)
        self.c_2 = np.float32(c_2)
        self.max_iteration_num = max_iteration_num
        self.min_value_change = min_value_change

        self.debug_wait_key = debug_wait_key
        
        self.particle_num = particle_num
        self.dimensions = len(self.lower_bounds)

        # self._setup_optimization() - add more members
    
    def _update_current_objective_values(self):
        self.current_particle_values = np.apply_along_axis(self.evaluator, 1, self.x, self.problem).astype(np.float32)
        
    def _update_particle_bests(self):
        improvement_mask = (self.current_particle_values < self.best_particle_values)
        self.best_particle_positions[improvement_mask, :] = self.x[improvement_mask, :].copy()
        self.best_particle_values[improvement_mask] = self.current_particle_values[improvement_mask]

    def _update_global_bests(self):
        argmin = np.argmin(self.best_particle_values)
        if self.best_particle_values[argmin] < self.best_global_value:
            self.best_global_value = self.best_particle_values[argmin]
            self.best_global_position = self.best_particle_positions[argmin, :].copy()

            if self.debug_wait_key is not None:
                drawer = HypergraphDrawer(self.problem, self.best_global_position)
                drawer.show(wait_key=self.debug_wait_key)
                print(self.best_global_value)
    
    def _update_velocities(self, w, c_1, c_2):
        r_1 = np.random.uniform(0, 1, size=(self.particle_num, self.dimensions))
        r_2 = np.random.uniform(0, 1, size=(self.particle_num, self.dimensions))
        self.velocities = w*self.velocities + c_1*r_1*(self.best_particle_positions - self.x) + c_2*r_2*(self.best_global_position - self.x)
    
    def _update_current_positions(self):
        self.x += self.velocities
        underflow_mask = self.x < self.lower_bounds
        overflow_mask = self.x > self.upper_bounds
        self.x = self.x*(~np.logical_or(underflow_mask, overflow_mask)) + self.lower_bounds*underflow_mask + self.upper_bounds*overflow_mask

    def _setup_optimization(self):
        bounds_range = self.upper_bounds - self.lower_bounds
        
        self.velocities = np.random.uniform(0, 1, size=(self.particle_num, self.dimensions))
        self.velocities = -bounds_range + self.lower_bounds + self.velocities * 2 * bounds_range # (bounds_range - -bounds_range) - max possible values - min possible values
        self.x = self.initializer(self.problem, self.particle_num) # current particle positions
        self.current_particle_values = np.zeros(self.particle_num, dtype=np.float32)
        self.best_particle_positions = np.zeros(self.x.shape, dtype=self.x.dtype)
        self.best_particle_values = np.full(self.x.shape[0], np.inf, dtype=np.float32)
        self.best_global_position = np.array([], dtype=self.x.dtype)
        self.best_global_value = np.inf
        
        self._update_current_objective_values()
        self._update_particle_bests()
        self._update_global_bests()

    def _run(self):
        iteration = 0
        while iteration < self.max_iteration_num:
            iteration += 1
            self._update_velocities(self.w, self.c_1, self.c_2)
            
            self._update_current_positions()
            self._update_current_objective_values()
            self._update_particle_bests()
            prev_best_global_value = self.best_global_value
            self._update_global_bests()
            
            if prev_best_global_value != self.best_global_value and np.abs(prev_best_global_value - self.best_global_value) <= self.min_value_change:
                return self.best_global_value, self.best_global_position, iteration
                
            if self.debug_wait_key is not None:
                print(iteration)

        return self.best_global_value, self.best_global_position, iteration

    def __call__(self):
        self._setup_optimization()
        return self._run()

class GA: # genetic Algorithm (instead of fitness value we use an objective function [lower value is better] here too)
    def __init__(self, problem, initializer, evaluator, population_size, selection_pct, max_iteration_num, min_value_change=1e-8, debug_wait_key=None):
        #assert int(selection_pct * population_size) * int(1.0 / selection_pct) == population_size, 'the given percentage would yield inconsistent population size'
        #pass
        assert int(population_size - population_size*selection_pct) % 2 == 0, 'the number of not selected population has to be even'

    def _selection(self):
        raise NotImplementedError

    def _crossover(self, selected_indices):
        raise NotImplementedError

    def _mutation(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError

class NaiveGA(GA):
    def __init__(self, problem, initializer, evaluator, population_size, selection_pct, mutation_random_generator, max_iteration_num, min_value_change=1e-8, debug_wait_key=None):
        super().__init__(problem, initializer, evaluator, population_size, selection_pct, max_iteration_num, min_value_change, debug_wait_key)
        self.evaluator = evaluator
        self.problem = problem
        self.lower_bounds = problem.get_vector_lower_bounds()
        self.upper_bounds = problem.get_vector_upper_bounds()
        self.initializer = initializer
        self.selection_pct = np.float32(selection_pct)
        self.mutation_random_generator = mutation_random_generator
        self.max_iteration_num = max_iteration_num
        self.min_value_change = min_value_change

        self.debug_wait_key = debug_wait_key
        
        self.population_size = population_size
        self.dimensions = len(problem.get_vector_lower_bounds())

    def _setup_optimization(self):
        self.x = self.initializer(self.problem, self.population_size) # current particle positions
        self.fitness_values = np.zeros(self.population_size, dtype=np.float32)
        self.best_position = np.array([], dtype=self.x.dtype)
        self.best_value = np.inf

        self._update()

    def _selection(self):
        num_selected = int(self.selection_pct * self.population_size)
        selected_indices = self.fitness_values.argsort()[:num_selected]
        return selected_indices

    def _crossover(self, selected_indices):
        def select_parent_ids(children_num):
            parent_1s = np.random.randint(0, self.population_size, size=(children_num//2,1)) # assertion is in place so children is always even
            parent_2s = np.random.randint(0, self.population_size - 1, size=(children_num//2,1))
            parent_2s[np.where(parent_1s <= parent_2s)] += 1
            parents = np.hstack((parent_1s, parent_2s))
            return parents
        
        def permute_node_ids(children_num):
            node_num = self.problem.hypergraph.shape[0]
            id_grid = np.tile(np.arange(node_num),(children_num//2, 1)) # assertion is in place so children is always even
            np.apply_along_axis(np.random.shuffle, 1, id_grid) # does it in place, so the columns of id_grid are shuffled now (row by row)
            return id_grid

        def combine_parents(parent_1_id, parent_2_id, node_id_permutation):
            node_num = len(node_id_permutation)
            parent_2_node_ids = node_id_permutation[-node_num//2:]
            parent_2_row_ids = np.hstack((parent_2_node_ids, parent_2_node_ids+node_num, parent_2_node_ids+node_num*2))
            new_row = self.x[parent_1_id, :].copy()
            new_row[parent_2_row_ids] = self.x[parent_2_id, parent_2_row_ids]
            return new_row
        
        first_children_wrapper = lambda combination_table_row : combine_parents(combination_table_row[0], combination_table_row[1], combination_table_row[2:])
        second_children_wrapper = lambda combination_table_row : combine_parents(combination_table_row[1], combination_table_row[0], combination_table_row[2:])

        children_num = self.population_size - len(selected_indices)
        #indices = np.arange(children_num).reshape(children_num, 1)
        combination_table = np.hstack((select_parent_ids(children_num), permute_node_ids(children_num)))
        first_children = np.apply_along_axis(first_children_wrapper, 1, combination_table)
        second_children = np.apply_along_axis(second_children_wrapper, 1, combination_table)
        best_values = self.x[selected_indices, :].copy()

        new_population = np.vstack((best_values, first_children, second_children))

        return new_population

    def _mutation(self, population, selected_indices):
        mutation = self.mutation_random_generator(size=population.shape)
        mutation[selected_indices, :] = 0
        population += mutation
        np.clip(population, self.lower_bounds, self.upper_bounds, out=population)
        return population

    def _update_fitness_values(self):
        self.fitness_values = np.apply_along_axis(self.evaluator, 1, self.x, self.problem).astype(np.float32)

    def _update_bests(self):
        argmin = np.argmin(self.fitness_values)
        if self.fitness_values[argmin] < self.best_value:
            self.best_value = self.fitness_values[argmin]
            self.best_position = self.x[argmin, :].copy()

            if self.debug_wait_key is not None:
                drawer = HypergraphDrawer(self.problem, self.best_position)
                drawer.show(wait_key=self.debug_wait_key)
                print(self.best_value)

    def _update(self):
        self._update_fitness_values()
        self._update_bests()

    def _run(self):
        iteration = 0
        while iteration < self.max_iteration_num:
            iteration += 1
            selected_indices = self._selection()
            new_population = self._crossover(selected_indices)
            self.x = self._mutation(new_population, selected_indices)
            self._update()

            if self.debug_wait_key is not None:
                print(iteration)

        return self.best_value, self.best_position, iteration

    def __call__(self):
        self._setup_optimization()
        return self._run()
