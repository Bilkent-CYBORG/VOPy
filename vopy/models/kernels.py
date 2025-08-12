from typing import Tuple, Optional

import torch

import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.models.kernels.categorical import CategoricalKernel


class MixedKernel(gpytorch.kernels.Kernel):
    """
    Kernel that combines a Matern kernel and a categorical (Hamming) kernel. It contains two
    components (one additive and one multiplicative) that are added together as follows:

    K(x1, x2) =
        K_cont_1(x1[cont], x2[cont]) + K_cat_1(x1[cat], x2[cat])
        + K_cont_2(x1[cont], x2[cont]) * K_cat_2(x1[cat], x2[cat])

    where x1[cont], x1[cont] are continuous inputs and x1[cat], x2[cat] are categorical inputs.

    :param continuous_dims: Indices of continuous dimensions (used for continuous kernel).
    :type continuous_dims: Tuple[int]
    :param categorical_dims: Indices of categorical dimensions (used for categorical kernel).
    :type categorical_dims: Tuple[int]
    :param continuous_kwargs: Additional keyword arguments for the continuous kernel.
    :type continuous_kwargs: Optional[dict]
    :param categorical_kwargs: Additional keyword arguments for the categorical kernel.
    :type categorical_kwargs: Optional[dict]
    """

    def __init__(
        self,
        continuous_dims: Tuple[int],
        categorical_dims: Tuple[int],
        continuous_kwargs: Optional[dict] = None,
        categorical_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.continuous_dims = continuous_dims
        self.categorical_dims = categorical_dims

        self.continuous_kernel_1 = self._build_cont_kernel(continuous_kwargs or {})
        self.continuous_kernel_2 = self._build_cont_kernel(continuous_kwargs or {})
        self.categorical_kernel_1 = self._build_cat_kernel(categorical_kwargs or {})
        self.categorical_kernel_2 = self._build_cat_kernel(categorical_kwargs or {})

    def _build_cont_kernel(self, opts: dict):
        """
        Builds a Matern kernel for continuous dimensions.

        :param opts: Options for the continuous kernel. It includes:
            - 'nu': The smoothness parameter for the Matern kernel (default is 5/2).
            - 'ard_num_dims': The number of dimensions for automatic relevance determination (ARD).
            - 'lengthscale_prior': Prior for the lengthscale (default is a GammaPrior(2, 3.0)).
            - 'outputscale_prior': Prior for the output scale.
        :type opts: dict
        :return: The specified continuous kernel.
        :rtype: gpytorch.kernels.Kernel
        """
        lengthscale_prior = opts.get("lengthscale_prior", gpytorch.priors.GammaPrior(2, 3.0))

        cont_base = MaternKernel(
            nu=opts.get("nu", 5 / 2),
            ard_num_dims=opts.get("ard_num_dims", len(self.continuous_dims)),
            active_dims=self.continuous_dims,
            batch_shape=self.batch_shape,
            lengthscale_prior=lengthscale_prior,
        )

        return ScaleKernel(
            cont_base,
            batch_shape=self.batch_shape,
            outputscale_prior=opts.get("outputscale_prior"),
        )

    def _build_cat_kernel(self, opts: dict):
        """
        Builds a categorical kernel for categorical dimensions.

        :param opts: Options for the categorical kernel. It includes:
            - 'ard_num_dims': The number of dimensions for automatic relevance determination (ARD).
            - 'outputscale_prior': Prior for the output scale.
        :type opts: dict
        :return: The specified categorical (Hamming) kernel.
        :rtype: gpytorch.kernels.Kernel
        """
        cat_base = CategoricalKernel(
            ard_num_dims=opts.get("ard_num_dims", len(self.categorical_dims)),
            active_dims=self.categorical_dims,
            batch_shape=self.batch_shape,
        )

        return ScaleKernel(
            cat_base,
            batch_shape=self.batch_shape,
            outputscale_prior=opts.get("outputscale_prior"),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **params):
        """
        Compute the kernel function as given in the class docstring.

        :param x1: First input tensor (continuous and categorical).
        :type x1: torch.Tensor
        :param x2: Second input tensor (continuous and categorical).
        :type x2: torch.Tensor
        :return: The computed kernel matrix. Shape is (batch_shape, n1, n2),
            where n1 and n2 are the number of elements in x1 and x2.
        :rtype: torch.Tensor
        """
        additive_term = self.continuous_kernel_1(x1, x2) + self.categorical_kernel_1(x1, x2)
        multiplicative_term = self.continuous_kernel_2(x1, x2) * self.categorical_kernel_2(x1, x2)
        return additive_term + multiplicative_term
