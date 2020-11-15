import numpy as np
from solution.drawing import HypergraphDrawer

class HypergraphPSO: # Particle Swarm Optimization
    def __init__(self, lower_bounds, upper_bounds, initializer, evaluator, particle_num, w, c_1, c_2, max_iteration_num, min_value_change=1e-8, debug=None, problem=None):
        # velocity_(T+1) = w*velocity_(T) + c_1*random_1*(best_particle_pos-current_pos) + c_2*random_2*(best_global_pos-current_pos)
        self.evaluator = evaluator
        self.problem = problem
        self.initializer = initializer
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.w = np.float32(w)
        self.c_1 = np.float32(c_1)
        self.c_2 = np.float32(c_2)
        self.max_iteration_num = max_iteration_num
        self.min_value_change = min_value_change

        self.debug = debug
        
        self.particle_num = particle_num
        self.dimensions = len(self.lower_bounds)

        # self._setup_optimization() - add more members
    
    def _update_current_objective_values(self):
        self.current_particle_values = np.apply_along_axis(self.evaluator, 1, self.x).astype(np.float32)[:,0]
        
    def _update_particle_bests(self):
        improvement_mask = (self.current_particle_values < self.best_particle_values)
        self.best_particle_positions[improvement_mask, :] = self.x[improvement_mask, :].copy()
        self.best_particle_values[improvement_mask] = self.current_particle_values[improvement_mask]

    def _show_debug_info(self):
       if self.debug is not None and self.problem is not None:
           drawer = HypergraphDrawer(self.problem, self.best_global_position)
           drawer.show(wait_key=self.debug)
           print(self.best_global_value)

    def _update_global_bests(self):
        argmin = np.argmin(self.best_particle_values)
        if self.best_particle_values[argmin] < self.best_global_value:
            self.best_global_value = self.best_particle_values[argmin]
            self.best_global_position = self.best_particle_positions[argmin, :].copy()

            self._show_debug_info()
    
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
        self.x = self.initializer() # current particle positions
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
                
            if self.debug is not None:
                print(iteration)

        return self.best_global_value, self.best_global_position, iteration

    def __call__(self):
        self._setup_optimization()
        return self._run()
