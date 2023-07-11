"""Implementation of Lagrange. copied from https://raw.githubusercontent.com/PKU-Alignment/omnisafe/main/omnisafe/common/lagrange.py"""

from __future__ import annotations
import flax

import optax
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState


class Lagrange(flax.struct.PyTreeNode):
    cost_limit: float = 0.0
    lambda_lr: float = 0
    lagrangian_upper_bound: float | None = 0
    init_value: float = 0.0
    state: TrainState = None

    @classmethod
    def create(cls, cost_limit, lambda_lr, lagrangian_upper_bound, init_value):
        lagrangian_multiplier = max(init_value, 0.0)

        trainstate = TrainState.create(
            apply_fn=None,
            params=lagrangian_multiplier,
            tx=optax.adam(learning_rate=lambda_lr),
        )

        return cls(
            cost_limit=cost_limit,
            lambda_lr=lambda_lr,
            lagrangian_upper_bound=lagrangian_upper_bound,
            init_value=init_value,
            state=trainstate,
        )

    @staticmethod
    def compute_lambda_loss(lagrangian_multiplier, mean_ep_cost: float, cost_limit):
        """Penalty loss for Lagrange multiplier.

        .. note::
            ``mean_ep_cost`` is obtained from ``self.logger.get_stats('EpCosts')[0]``, which is
            already averaged across MPI processes.

        Args:
            mean_ep_cost (float): mean episode cost.

        Returns:
            Penalty loss for Lagrange multiplier.
        """
        return -lagrangian_multiplier * (mean_ep_cost - cost_limit)

    def update_lagrange_multiplier(self, Jc: float):
        r"""Update Lagrange multiplier (lambda).

        We update the Lagrange multiplier by minimizing the penalty loss, which is defined as:

        .. math::

            \lambda ^{'} = \lambda + \eta \cdot (J_C - J_C^*)

        where :math:`\lambda` is the Lagrange multiplier, :math:`\eta` is the learning rate,
        :math:`J_C` is the mean episode cost, and :math:`J_C^*` is the cost limit.

        Args:
            Jc (float): mean episode cost.
        """

        loss_value, grads = jax.value_and_grad(self.compute_lambda_loss)(
            self.state.params, Jc, self.cost_limit
        )
        state = self.state.apply_gradients(grads=grads)
        return self.replace(state=state.replace(params=jnp.clip(state.params, 0)))
