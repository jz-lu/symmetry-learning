import torch

class GradientDescent:
    def __init__(self, maxiter, learning_rate, perturbation = 0.01):
        self.maxiter = maxiter
        self.eta = learning_rate
        self.perturbation = perturbation

    def optimize(self, num_vars, objective_function, initial_point, variable_bounds):
        param = initial_point
        print("sanity")
        for iter in range(self.maxiter):
            cur_val = objective_function(param)
            grad = torch.zeros(num_vars)
            for i in range(num_vars):
                new_param = param.clone()
                new_param[i] = new_param[i] + self.perturbation
                new_val = objective_function(new_param)
                grad[i] = (new_val - cur_val) / self.perturbation
            param = param - self.eta * grad
        return param, objective_function(param), self.maxiter
