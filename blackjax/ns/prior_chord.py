# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Nested slice sampling with exact unit-cube chord brackets."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.slice import SliceInfo, _shrink
from blackjax.ns import adaptive, from_mcmc, nss
from blackjax.ns.base import init_state_strategy
from blackjax.types import Array

__all__ = ["as_top_level_api"]


def _unit_chord_bounds(position: Array, direction: Array) -> tuple[Array, Array]:
    """Find the intersection of a line through the current point and the cube."""
    nonzero = direction != 0
    safe_direction = jnp.where(nonzero, direction, 1)
    at_zero = -position / safe_direction
    at_one = (1 - position) / safe_direction
    lower = jnp.where(nonzero, jnp.minimum(at_zero, at_one), -jnp.inf)
    upper = jnp.where(nonzero, jnp.maximum(at_zero, at_one), jnp.inf)
    return jnp.max(lower), jnp.min(upper)


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_inner_steps: int,
    *,
    num_delete: int = 1,
    max_shrinkage: int = 100,
) -> SamplingAlgorithm:
    """Build nested slice sampling on a uniform unit cube.

    Live positions are floating-point vectors in ``[0, 1]^D``. The prior
    function is constant on its allowed support and ``-inf`` outside it.
    To sample a nonuniform physical prior, apply its prior transform inside
    ``loglikelihood_fn``. The likelihood must be finite wherever it is
    nonzero; ``-inf`` denotes zero likelihood.

    Each inner step draws a covariance-shaped direction and shrinks the
    intersection of its line and the unit cube until a point obeys the
    prior support and exceeds the likelihood contour. It needs no stepping
    out. If the shrinkage budget is exhausted, the chain stays in place.
    """
    init_particle = partial(
        init_state_strategy, logprior_fn=logprior_fn, loglikelihood_fn=loglikelihood_fn
    )

    def move(rng_key, state, loglikelihood_0, covariance_factor):
        direction_key, shrink_key = jax.random.split(rng_key)
        direction = nss.sample_direction_from_covariance_factor(
            direction_key, state.position, covariance_factor
        )
        left, right = _unit_chord_bounds(state.position, direction)

        def slice_fn(t):
            proposed = init_particle(
                state.position + t * direction,
                loglikelihood_birth=loglikelihood_0,
            )
            in_slice = (
                jnp.isfinite(proposed.logdensity)
                & jnp.isfinite(proposed.loglikelihood)
                & (proposed.loglikelihood > loglikelihood_0)
            )
            return proposed, in_slice

        new_state, num_shrink, accepted = _shrink(
            shrink_key,
            slice_fn,
            -jnp.inf,
            lambda _: jnp.asarray(True),
            left,
            right,
            state,
            max_shrinkage,
        )
        return new_state, SliceInfo(accepted, jnp.asarray(0), num_shrink, left, right)

    kernel = from_mcmc.build_kernel(
        move, num_inner_steps, nss.live_covariance_factor, num_delete
    )

    def init(position, rng_key=None):
        return adaptive.init(
            position,
            init_state_fn=jax.vmap(init_particle),
            update_inner_kernel_params_fn=nss.live_covariance_factor,
            rng_key=rng_key,
        )

    return SamplingAlgorithm(init, kernel)
