import logging
from abc import ABC
from typing import Optional, List, Union

import torch
import gpytorch
from gpytorch.mlls import SumMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

import numpy as np

from vectoptal.models import GPModel, ModelList
from vectoptal.maximization_problem import Problem
from vectoptal.utils.utils import generate_sobol_samples

torch.set_default_dtype(torch.float64)


class GPyTorchModel(GPModel, ABC):
    """
    Base class for Gaussian Process models using GPyTorch. Provides
    utility methods for data handling and kernel type identification.
    """

    def __init__(self) -> None:
        super().__init__()

    def to_tensor(self, data) -> torch.Tensor:
        """
        Converts input data to a PyTorch tensor.

        :param data: Input data to convert to tensor format.
        :type data: array-like or torch.Tensor
        :return: Converted PyTorch tensor.
        :rtype: torch.Tensor
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float64).to(self.device)

        return data

    def get_kernel_type(self):
        """
        Identifies the type of kernel used in the model, such as RBF or Multitask.

        :return: Kernel type as a string ("RBF", "Multitask", or "Other").
        :rtype: str
        """
        kernel = self.model.covar_module
        while True:
            if isinstance(kernel, gpytorch.kernels.RBFKernel):
                return "RBF"
            elif isinstance(kernel, gpytorch.kernels.MultitaskKernel):
                kernel = kernel.data_covar_module
                continue
            elif hasattr(kernel, "base_kernel"):
                kernel = kernel.base_kernel
                continue
            else:
                return "Other"


# TODO: Make GPyTorch models optionally take covar_module and mean_module.
class MultitaskExactGPModel(gpytorch.models.ExactGP):
    """
    Exact GP model for multitask problems with dependent objectives.

    :param train_inputs: Input training data.
    :type train_inputs: torch.Tensor
    :param train_targets: Target training data.
    :type train_targets: torch.Tensor
    :param likelihood: Gaussian likelihood module.
    :type likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood
    :param kernel: Kernel type for covariance computation.
    :type kernel: gpytorch.kernels.Kernel
    """

    def __init__(self, train_inputs, train_targets, likelihood, kernel):
        super().__init__(train_inputs, train_targets, likelihood)

        input_dim = train_inputs.shape[-1]
        output_dim = train_targets.shape[-1]

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=output_dim
        )

        # self.base_covar_module = gpytorch.kernels.ScaleKernel(kernel(ard_num_dims=input_dim))
        self.base_covar_module = kernel(
            ard_num_dims=input_dim, lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0)
        )  # Scale kernel is unnecessary
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            self.base_covar_module, num_tasks=output_dim, rank=output_dim
        )

        self.mean_module.requires_grad_(True)
        self.covar_module.requires_grad_(True)

    def forward(self, x):
        """
        Computes the mean and covariance.

        :param x: Input data.
        :type x: torch.Tensor
        :return: MultitaskMultivariateNormal distribution with mean and covariance.
        :rtype: gpytorch.distributions.MultitaskMultivariateNormal
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class BatchIndependentExactGPModel(gpytorch.models.ExactGP):
    """
    Exact GP model for multitask problems with independent objectives.

    :param train_inputs: Input training data.
    :type train_inputs: torch.Tensor
    :param train_targets: Target training data.
    :type train_targets: torch.Tensor
    :param likelihood: Gaussian likelihood module.
    :type likelihood: gpytorch.likelihoods.GaussianLikelihood
    :param kernel: Kernel type for covariance computation.
    :type kernel: gpytorch.kernels.Kernel
    """

    def __init__(self, train_inputs, train_targets, likelihood, kernel):
        super().__init__(train_inputs, train_targets, likelihood)

        input_dim = train_inputs.shape[-1]
        output_dim = train_targets.shape[-1]

        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([output_dim]))

        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernel(batch_shape=torch.Size([output_dim]), ard_num_dims=input_dim),
            batch_shape=torch.Size([output_dim]),
        )

        self.mean_module.requires_grad_(True)
        self.covar_module.requires_grad_(True)

    def forward(self, x):
        """
        Computes the mean and covariance.

        :param x: Input data.
        :type x: torch.Tensor
        :return: MultitaskMultivariateNormal distribution with mean and covariance.
        :rtype: gpytorch.distributions.MultitaskMultivariateNormal
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class GPyTorchMultioutputExactModel(GPyTorchModel, ABC):
    """
    Multioutput GP model with exact inference, assuming known noise variance.

    :param input_dim: Dimensionality of the input data.
    :type input_dim: int
    :param output_dim: Dimensionality of the output data.
    :type output_dim: int
    :param noise_var: Noise variance for Gaussian likelihood.
    :type noise_var: float
    :param model_kind: Type of Exact GP model to be used.
    :type model_kind: gpytorch.models.ExactGP

    Methods
    -------
    add_sample(X_t, Y_t)
        Adds a new sample to the training data.
    clear_data()
        Clears stored training data.
    update()
        Updates the GP model with the current training data.
    train()
        Trains the GP model using maximum likelihood estimation.
    evaluate_kernel(X=None)
        Evaluates the kernel for given data points.
    get_lengthscale_and_var()
        Returns the lengthscale and variance of the kernel.
    """

    def __init__(
        self, input_dim, output_dim, noise_var, model_kind: gpytorch.models.ExactGP
    ) -> None:
        super().__init__()

        self.device = "cpu"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_kind = model_kind

        # Data containers.
        self.clear_data()

        # Set up likelihood
        self.noise_var = torch.tensor(noise_var)
        if self.noise_var.dim() > 1:
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=self.output_dim,
                rank=len(self.noise_var),
                noise_constraint=gpytorch.constraints.GreaterThan(1e-10),
                has_global_noise=False,
            ).to(self.device)
            self.likelihood.task_noise_covar = self.noise_var
        else:
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=self.output_dim,
                rank=0,
                noise_constraint=gpytorch.constraints.GreaterThan(1e-10),
                has_task_noise=False,
            ).to(self.device)
            self.likelihood.noise = self.noise_var
        self.likelihood.requires_grad_(False)

        self.kernel_type = gpytorch.kernels.RBFKernel
        self.model = None

    def add_sample(self, X_t, Y_t):
        """
        Adds a new sample to the training data.

        :param X_t: Input data sample.
        :type X_t: torch.Tensor
        :param Y_t: Target data sample.
        :type Y_t: torch.Tensor
        """
        X_t = self.to_tensor(X_t[..., : self.input_dim])
        Y_t = self.to_tensor(Y_t)

        assert X_t.ndim == 2 and Y_t.ndim == 2, (
            "Model expects a 2D and 1D tensors for X_t and Y_t,"
            f"but got {X_t.ndim} and {Y_t.ndim} instead."
        )

        self.train_inputs = torch.cat([self.train_inputs, X_t], 0)
        self.train_targets = torch.cat([self.train_targets, Y_t], 0)

    def clear_data(self):
        """
        Clears stored training data.
        """
        self.train_inputs = torch.empty((0, self.input_dim)).to(self.device)
        self.train_targets = torch.empty((0, self.output_dim)).to(self.device)

    def update(self):
        """
        Updates the GP model with the current training data.
        """
        if self.model is None:
            self.model = self.model_kind(
                self.train_inputs, self.train_targets, self.likelihood, self.kernel_type
            ).to(self.device)
        else:
            self.model.set_train_data(self.train_inputs, self.train_targets, strict=False)

        self.model.eval()
        self.likelihood.eval()

    def train(self):
        """
        Trains the GP model.
        """

        self.model.train()
        self.likelihood.train()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        logging.info("Training started.")
        fit_gpytorch_mll(mll)
        logging.info("Training done.")

        self.model.eval()
        self.likelihood.eval()

    def evaluate_kernel(self, X=None):
        """
        Evaluates the kernel for given data points.

        :param X: Input data to evaluate. Defaults to training data if None.
        :type X: torch.Tensor, optional
        :return: Kernel matrix.
        :rtype: np.ndarray
        """

        assert self.model is not None, "Kernel evaluated before model initialization."

        if X is None:
            X = self.train_inputs
        Kn = self.model.covar_module(X, X).evaluate()
        Kn = Kn.numpy(force=True)

        return Kn

    def get_lengthscale_and_var(self):
        """
        Returns the kernel lengthscale and variance.

        :return: Lengthscales and variances as arrays.
        :rtype: tuple (np.ndarray, np.ndarray)
        """
        assert self.model is not None, "Model not initialized."

        cov_module = self.model.covar_module
        if isinstance(self.model, MultitaskExactGPModel):
            lengthscales = cov_module.data_covar_module.lengthscale.squeeze().numpy(force=True)
            variances = cov_module.task_covar_module.var.squeeze().numpy(force=True)
        elif isinstance(self.model, BatchIndependentExactGPModel):
            lengthscales = cov_module.base_kernel.lengthscale.squeeze().numpy(force=True)
            variances = cov_module.outputscale.squeeze().numpy(force=True)

        return lengthscales, variances


class CorrelatedExactGPyTorchModel(GPyTorchMultioutputExactModel):
    """
    Correlated multitask GP model using the multioutput exact GP framework.

    :param input_dim: Dimensionality of the input data.
    :type input_dim: int
    :param output_dim: Dimensionality of the output data.
    :type output_dim: int
    :param noise_var: Noise variance for Gaussian likelihood.
    :type noise_var: float
    """

    def __init__(self, input_dim, output_dim, noise_var) -> None:
        super().__init__(input_dim, output_dim, noise_var, MultitaskExactGPModel)

    def predict(self, test_X):
        """
        Makes predictions on test data.

        :param test_X: Test data.
        :type test_X: torch.Tensor
        :return: Predicted means and variances.
        :rtype: tuple (np.ndarray, np.ndarray)
        """
        # Last column of X_t are sample space indices.
        test_X = self.to_tensor(test_X[:, : self.input_dim])

        test_X = test_X[:, None, :]  # Prepare for batch inference

        with torch.no_grad(), torch.autograd.set_detect_anomaly(True):
            res = self.model(test_X)

            means = res.mean.squeeze().cpu().numpy()  # Squeeze the sample dimension
            variances = res.covariance_matrix
            variances = variances.cpu().numpy()

        return means, variances


class IndependentExactGPyTorchModel(GPyTorchMultioutputExactModel):
    """
    Independent multitask GP model using the multioutput exact GP framework.

    :param input_dim: Dimensionality of the input data.
    :type input_dim: int
    :param output_dim: Dimensionality of the output data.
    :type output_dim: int
    :param noise_var: Noise variance for Gaussian likelihood.
    :type noise_var: float
    """

    def __init__(self, input_dim, output_dim, noise_var) -> None:
        super().__init__(input_dim, output_dim, noise_var, BatchIndependentExactGPModel)

    def predict(self, test_X):
        """
        Makes predictions on test data.

        :param test_X: Test data.
        :type test_X: torch.Tensor
        :return: Predicted means and variances.
        :rtype: tuple (np.ndarray, np.ndarray)
        """
        # Last column of X_t are sample space indices.
        test_X = self.to_tensor(test_X[..., : self.input_dim])

        with torch.no_grad(), torch.autograd.set_detect_anomaly(True):
            res = self.model(test_X)

            means = res.mean.squeeze().cpu().numpy()  # Squeeze the sample dimension
            variances = (
                torch.einsum("ij,ki->kij", torch.eye(self.output_dim), res.variance).cpu().numpy()
            )

        return means, variances


def get_gpytorch_model_w_known_hyperparams(
    model_class,
    problem: Problem,
    noise_var: float,
    initial_sample_cnt: int,
    X: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
) -> GPyTorchMultioutputExactModel:
    """
    Creates and returns a GPyTorch model after training and freezing model parameters.
    If X and Y is not given, sobol samples are evaluated to generate a learning dataset.

    :param model_class: Class of the GP model to instantiate.
    :type model_class: type
    :param problem: Problem instance defining the optimization problem.
    :type problem: Problem
    :param noise_var: Noise variance for Gaussian likelihood.
    :type noise_var: float
    :param initial_sample_cnt: Number of initial samples to use for training.
    :type initial_sample_cnt: int
    :param X: Optional initial input data for training.
    :type X: np.ndarray, optional
    :param Y: Optional initial target data for training.
    :type Y: np.ndarray, optional
    :return: Trained GP model instance.
    :rtype: GPyTorchMultioutputExactModel
    """
    if X is None:
        X = generate_sobol_samples(problem.in_dim, 1024)  # TODO: magic number
    if Y is None:
        Y = problem.evaluate(X)

    in_dim = X.shape[1]
    out_dim = Y.shape[1]

    model = model_class(in_dim, out_dim, noise_var=noise_var)

    model.add_sample(X, Y)
    model.update()
    model.train()
    model.clear_data()

    # TODO: Initial sampling should be done outside of here. Can be a utility function.
    if initial_sample_cnt > 0:
        initial_indices = np.random.choice(len(X), initial_sample_cnt)
        initial_points = X[initial_indices]
        initial_values = Y[initial_indices]

        model.add_sample(initial_points, initial_values)
        model.update()

    return model


class SingleTaskGP(gpytorch.models.ExactGP):
    """
    Single-task Gaussian Process (GP) model with exact inference.

    :param train_inputs: Training input data.
    :type train_inputs: torch.Tensor
    :param train_targets: Training target data.
    :type train_targets: torch.Tensor
    :param likelihood: Gaussian likelihood module.
    :type likelihood: gpytorch.likelihoods.GaussianLikelihood
    :param kernel: Kernel function for the covariance module.
    :type kernel: gpytorch.kernels.Kernel
    """

    def __init__(self, train_inputs, train_targets, likelihood, kernel):
        super().__init__(train_inputs, train_targets, likelihood)

        input_dim = train_inputs.shape[-1]

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel(ard_num_dims=input_dim))

        self.mean_module.requires_grad_(True)
        self.covar_module.requires_grad_(True)

    def forward(self, x):
        """
        Computes the forward pass for the single-task GP model.

        :param x: Input data.
        :type x: torch.Tensor
        :return: MultivariateNormal distribution with computed mean and covariance.
        :rtype: gpytorch.distributions.MultivariateNormal
        """
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class GPyTorchModelListExactModel(GPyTorchModel, ModelList):
    """
    Multi-output GP model implemented as a list of independent single-task GP models.

    :param input_dim: Dimensionality of the input data.
    :type input_dim: int
    :param output_dim: Dimensionality of the output data.
    :type output_dim: int
    :param noise_var: Noise variance for the Gaussian likelihood.
    :type noise_var: float
    """

    def __init__(self, input_dim, output_dim, noise_var) -> None:
        super().__init__()

        self.device = "cpu"

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Data containers.
        self.clear_data()

        # Set up likelihood
        self.noise_var = torch.tensor(noise_var)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-10),
        ).to(self.device)
        likelihood.noise = self.noise_var
        likelihood.requires_grad_(False)
        self.likelihoods = [likelihood] * self.output_dim

        self.kernel_type = gpytorch.kernels.RBFKernel
        self.model = None

    def _add_sample_single(self, X_t, Y_t, dim_index: int):
        """
        Adds a new sample to the training data for specified output dimension(s).

        :param X_t: Input sample data.
        :type X_t: torch.Tensor
        :param Y_t: Target sample data.
        :type Y_t: torch.Tensor
        :param dim_index: Index or indices of output dimensions to update.
        :type dim_index: int or list of int
        """
        self.train_inputs[dim_index] = torch.cat([self.train_inputs[dim_index], X_t], 0)
        self.train_targets[dim_index] = torch.cat([self.train_targets[dim_index], Y_t], 0)

    def add_sample(self, X_t, Y_t, dim_index: Union[int, List[int]]):
        """
        Adds a new sample to the training data for specified output dimension(s).

        :param X_t: Input sample data.
        :type X_t: torch.Tensor
        :param Y_t: Target sample data.
        :type Y_t: torch.Tensor
        :param dim_index: Index or indices of output dimensions to update.
        :type dim_index: int or list of int
        """
        X_t = self.to_tensor(X_t[..., : self.input_dim])
        Y_t = self.to_tensor(Y_t)

        assert X_t.ndim == 2 and Y_t.ndim == 1, (
            "Model expects a 2D and 1D tensors for X_t and Y_t,"
            f"but got {X_t.ndim} and {Y_t.ndim} instead."
        )

        # All samples belongs to same model
        if isinstance(dim_index, int):
            self._add_sample_single(X_t, Y_t, dim_index)
            return

        # Each sample is for corresponding model
        assert len(dim_index) == len(X_t), "dim_index should be the same length as data"

        dim_index = torch.tensor(dim_index, dtype=torch.int32)
        unique_dims = torch.unique(dim_index)
        for dim_i in unique_dims:
            sample_indices_dim = dim_index == dim_i
            self._add_sample_single(
                X_t[sample_indices_dim].reshape(-1, self.input_dim), Y_t[sample_indices_dim], dim_i
            )

    def clear_data(self):
        """
        Clears all stored training data.
        """
        self.train_inputs = [
            torch.empty((0, self.input_dim)).to(self.device) for _ in range(self.output_dim)
        ]
        self.train_targets = [torch.empty((0)).to(self.device) for _ in range(self.output_dim)]

    def update(self):
        """
        Updates the model with the current training data.
        Initializes the model if not already set.
        """
        if self.model is None:
            models = [
                SingleTaskGP(
                    self.train_inputs[obj_i],
                    self.train_targets[obj_i],
                    self.likelihoods[obj_i],
                    kernel=self.kernel_type,
                )
                for obj_i in range(self.output_dim)
            ]
            self.model = gpytorch.models.IndependentModelList(*models)
            self.likelihood = gpytorch.likelihoods.LikelihoodList(
                *[model.likelihood for model in self.model.models]
            )
        else:
            for obj_i in range(self.output_dim):
                self.model.models[obj_i].set_train_data(
                    self.train_inputs[obj_i], self.train_targets[obj_i], strict=False
                )

        self.model.eval()
        self.likelihood.eval()

    def train(self):
        """
        Trains the model.
        """

        self.model.train()
        self.likelihood.train()

        mll = SumMarginalLogLikelihood(self.likelihood, self.model)

        logging.info("Training started.")
        fit_gpytorch_mll(mll)
        logging.info("Training done.")

        self.model.eval()
        self.likelihood.eval()

    def evaluate_kernel(self, X):
        """
        Evaluates the covariance kernel for the input data across all output dimensions.

        :param X: M x dim input data points for kernel evaluation.
        :type X: torch.Tensor
        :return: Covariance matrix of shape NM x NM, where N is the number of models, i.e. output dimension.
        :rtype: np.ndarray
        """
        N = len(self.model.models)
        M = len(X)

        Kn_i = torch.stack([model.covar_module(X, X).evaluate() for model in self.model.models])
        Kn = Kn_i.unsqueeze(2).expand(-1, -1, N, -1).reshape(N * M, N * M)
        Kn = Kn.numpy(force=True)

        return Kn

    def get_lengthscale_and_var(self):
        """
        Retrieves the kernel lengthscale and variance for each model (output dimension).

        :return: Lengthscales and variances for each output dimension.
        :rtype: tuple (np.ndarray, np.ndarray)
        """
        lengthscales = np.zeros(self.input_dim)
        variances = np.zeros(self.input_dim)
        for model_i, model in enumerate(self.model.models):
            cov_module = model.covar_module
            lengthscale = cov_module.base_kernel.lengthscale.squeeze().numpy(force=True).item()
            variance = cov_module.outputscale.squeeze().numpy(force=True).item()

            lengthscales[model_i] = lengthscale
            variances[model_i] = variance

        return lengthscales, variances

    def predict(self, test_X):
        """
        Makes predictions on test data across all output dimensions.

        :param test_X: Test data input.
        :type test_X: torch.Tensor
        :return: Predicted means and variances for each output dimension.
        :rtype: tuple (np.ndarray, np.ndarray)
        """
        test_X = self.to_tensor(test_X)

        test_X = test_X[:, None, :]  # Prepare for batch inference

        means = np.zeros((len(test_X), self.output_dim))
        variances = np.zeros((len(test_X), self.output_dim, self.output_dim))

        predictions: List[gpytorch.distributions.MultivariateNormal] = self.model(
            *[test_X for _ in range(self.output_dim)]
        )
        for dim_i in range(self.output_dim):
            mean = predictions[dim_i].mean.squeeze().numpy(force=True)
            var = predictions[dim_i].covariance_matrix.squeeze().numpy(force=True)
            means[:, dim_i] = mean.copy()
            variances[:, dim_i, dim_i] = var

        return means, variances

    def sample_from_posterior(self, test_X, sample_count=1):
        """
        Samples from the posterior distribution of all output dimensions.

        :param test_X: Test data input.
        :type test_X: torch.Tensor
        :param sample_count: Number of samples to draw.
        :type sample_count: int
        :return: Samples from the posterior distribution.
        :rtype: np.ndarray
        """
        test_X = self.to_tensor(test_X)

        # Generate MultivariateNormals from each model
        predictions: List[gpytorch.distributions.MultivariateNormal] = self.model(
            *[test_X for _ in range(self.output_dim)]
        )

        # Combine them into a MultitaskMultivariateNormal
        posterior = gpytorch.distributions.MultitaskMultivariateNormal.from_independent_mvns(
            predictions
        )

        samples = posterior.sample(torch.Size([sample_count])).numpy(force=True)

        return samples

    def sample_from_single_posterior(self, test_X, dim_index: int, sample_count=1):
        """
        Samples from the posterior distribution of a single output dimension.

        :param test_X: Test data input.
        :type test_X: torch.Tensor
        :param dim_index: Index of the output dimension.
        :type dim_index: int
        :param sample_count: Number of samples to draw.
        :type sample_count: int
        :return: Samples from the posterior distribution of the specified dimension.
        :rtype: np.ndarray
        """
        test_X = self.to_tensor(test_X)

        # Generate MultivariateNormal from the model
        posterior: gpytorch.distributions.MultivariateNormal = self.model.models[dim_index](test_X)

        samples = posterior.sample(torch.Size([sample_count])).numpy(force=True)

        return samples


def get_gpytorch_modellist_w_known_hyperparams(
    problem: Problem,
    noise_var: float,
    initial_sample_cnt: int,
    X: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
) -> GPyTorchModelListExactModel:
    """
    Creates and returns a GPyTorch model list after training and freezing model parameters.
    If `X` and `Y` is not given, sobol samples are evaluated to generate a learning dataset.

    :param problem: An instance of the optimization problem.
    :type problem: Problem
    :param noise_var: Noise variance for Gaussian likelihood.
    :type noise_var: float
    :param initial_sample_cnt: Number of samples to initially use for training.
    :type initial_sample_cnt: int
    :param X: Initial input data for training, if available.
    :type X: np.ndarray, optional
    :param Y: Initial target data for training, if available.
    :type Y: np.ndarray, optional
    :return: Trained multi-output GP model list.
    :rtype: GPyTorchModelListExactModel
    """
    if X is None:
        X = generate_sobol_samples(problem.in_dim, 1024)  # TODO: magic number
    if Y is None:
        Y = problem.evaluate(X)

    in_dim = X.shape[1]
    out_dim = Y.shape[1]

    model = GPyTorchModelListExactModel(in_dim, out_dim, noise_var=noise_var)

    for dim_i in range(out_dim):
        model.add_sample(X, Y[:, dim_i], dim_i)
    model.update()
    model.train()
    model.clear_data()

    # TODO: Initial sampling should be done outside of here. Can be a utility function.
    if initial_sample_cnt > 0:
        choices = np.stack(
            [  # Expand X to also include dimensions
                np.arange(len(X)).repeat(out_dim),
                np.tile(np.arange(out_dim), len(X)),
            ],
            axis=-1,
        )
        selected_pt_obj_indices = np.random.choice(len(choices), initial_sample_cnt)
        initial_pt_obj_indices = choices[selected_pt_obj_indices]
        initial_points = X[initial_pt_obj_indices[:, 0]]
        initial_values = problem.evaluate(initial_points)

        model.add_sample(
            initial_points,
            initial_values[np.arange(initial_sample_cnt), initial_pt_obj_indices[:, 1]],
            initial_pt_obj_indices[:, 1],
        )
        model.update()

    return model
