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
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import random

from blackjax.base import SamplingAlgorithm
from blackjax.ns import adaptive, from_mcmc, nss
from blackjax.ns.base import init_state_strategy

__all__ = [
    "PriorChordInfo",
    "ReplacementUpdateInfo",
    "unit_chord_bounds",
    "prior_chord_slice",
    "replacement_update_with_mcmc_take_last",
    "build_prior_chord_algorithm",
]


class PriorChordInfo(NamedTuple):
    """Diagnostics for one prior-chord slice move."""

    is_accepted: jax.Array
    num_expansions: jax.Array
    num_shrink: jax.Array
    bracket_left: jax.Array
    bracket_right: jax.Array
    has_invalid_loglikelihood: jax.Array
    start_loglikelihood: jax.Array
    contour_loglikelihood: jax.Array


class ReplacementUpdateInfo(NamedTuple):
    """Diagnostics for each move, each chain, and the number of survivors."""

    is_accepted: jax.Array
    num_expansions: jax.Array
    num_shrink: jax.Array
    bracket_left: jax.Array
    bracket_right: jax.Array
    has_invalid_loglikelihood: jax.Array
    start_loglikelihood: jax.Array
    contour_loglikelihood: jax.Array
    parent_index: jax.Array
    survivor_count: jax.Array
    start_to_final_distance: jax.Array
    start_chain_loglikelihood: jax.Array
    final_chain_loglikelihood: jax.Array


def unit_chord_bounds(position, direction) -> tuple[jax.Array, jax.Array]:
    """Return the parameters of the line segment inside ``[0, 1]^D``.

    The position lies in the cube and the direction is nonzero. A zero
    component of the direction imposes no constraint along that axis.
    """
    x = jnp.asarray(position)
    d = jnp.asarray(direction)
    zero = jnp.asarray(0.0, dtype=x.dtype)
    one = jnp.asarray(1.0, dtype=x.dtype)
    inf = jnp.asarray(jnp.inf, dtype=x.dtype)

    safe_d = jnp.where(d == 0.0, one, d)
    lower_positive = -x / safe_d
    upper_positive = (one - x) / safe_d
    lower_negative = (one - x) / safe_d
    upper_negative = -x / safe_d
    lower = jnp.where(
        d > zero,
        lower_positive,
        jnp.where(d < zero, lower_negative, -inf),
    )
    upper = jnp.where(
        d > zero,
        upper_positive,
        jnp.where(d < zero, upper_negative, inf),
    )
    return jnp.max(lower), jnp.min(upper)


def prior_chord_slice(
    rng_key,
    state,
    init_state_fn: Callable,
    direction,
    loglikelihood_0,
    max_shrinkage: int,
):
    """Shrink the exact cube chord until a point is in prior and contour support."""
    left, right = unit_chord_bounds(state.position, direction)
    zero_i = jnp.asarray(0, dtype=jnp.int32)
    birth = loglikelihood_0 * jnp.ones_like(state.loglikelihood_birth)
    birth_state = state._replace(loglikelihood_birth=birth)

    def cond(carry):
        n, _, _, _, _, accepted, _ = carry
        return (n < max_shrinkage) & ~accepted

    def body(carry):
        n, key, current_left, current_right, candidate_state, _, invalid = carry
        key, proposal_key = random.split(key)
        t = random.uniform(proposal_key, minval=current_left, maxval=current_right)
        position = jax.tree.map(lambda x, d: x + t * d, state.position, direction)
        proposed = init_state_fn(position, loglikelihood_birth=loglikelihood_0)
        invalid_proposal = jnp.isnan(proposed.loglikelihood) | jnp.isposinf(
            proposed.loglikelihood
        )
        invalid = invalid | invalid_proposal
        accepted = (
            jnp.isfinite(proposed.logdensity)
            & jnp.isfinite(proposed.loglikelihood)
            & (proposed.loglikelihood > loglikelihood_0)
        )
        current_left = jnp.where(t < 0.0, t, current_left)
        current_right = jnp.where(t >= 0.0, t, current_right)
        candidate_state = jax.tree.map(
            lambda new, old: jnp.where(accepted, new, old), proposed, candidate_state
        )
        return (
            n + 1,
            key,
            current_left,
            current_right,
            candidate_state,
            accepted,
            invalid,
        )

    initial = (
        zero_i,
        rng_key,
        left,
        right,
        birth_state,
        jnp.asarray(False),
        jnp.asarray(False),
    )
    n, _, _, _, candidate, accepted, invalid = jax.lax.while_loop(cond, body, initial)
    return candidate, PriorChordInfo(
        is_accepted=accepted,
        num_expansions=zero_i,
        num_shrink=n,
        bracket_left=left,
        bracket_right=right,
        has_invalid_loglikelihood=invalid,
        start_loglikelihood=state.loglikelihood,
        contour_loglikelihood=loglikelihood_0,
    )


def replacement_update_with_mcmc_take_last(
    constrained_step_fn: Callable,
    num_mcmc_steps: int,
    num_delete: int,
    replacement_chunks: int,
):
    """Start chains from strict survivors and run them in equal sized batches.

    The step must return ``PriorChordInfo``. A zero survivor count means
    replacement was impossible and the caller must stop; the output state is
    not usable in that case.
    """
    if replacement_chunks < 1:
        raise ValueError("replacement_chunks must be positive")
    if num_delete % replacement_chunks:
        raise ValueError("num_delete must be divisible by replacement_chunks")

    def update_function(rng_key, state, loglikelihood_0, **step_parameters):
        choice_key, sample_key = random.split(rng_key)
        particles = state.particles
        survivor_mask = particles.loglikelihood > loglikelihood_0
        survivor_count = jnp.sum(survivor_mask, dtype=jnp.int32)
        survivor_idx = jnp.nonzero(
            survivor_mask, size=len(particles.loglikelihood), fill_value=0
        )[0]
        # Compaction prevents zero-weight particles from being selected.
        # A count of zero uses a placeholder index to keep the array program
        # defined until its caller can report the invalid contour.
        start_rank = random.randint(
            choice_key,
            shape=(num_delete,),
            minval=0,
            maxval=jnp.maximum(survivor_count, 1),
        )
        start_idx = survivor_idx[start_rank]
        start_state = jax.tree.map(lambda x: x[start_idx], particles)
        shared_step = partial(
            constrained_step_fn, loglikelihood_0=loglikelihood_0, **step_parameters
        )
        new_particles, move_info = from_mcmc._run_mcmc_chains(
            shared_step,
            num_mcmc_steps,
            random.split(sample_key, num_delete),
            start_state,
            replacement_chunks,
        )

        distance_dtype = jax.dtypes.canonicalize_dtype(jnp.float64)
        squared_distance = jnp.zeros(num_delete, dtype=distance_dtype)
        for initial_leaf, final_leaf in zip(
            jax.tree.leaves(start_state.position),
            jax.tree.leaves(new_particles.position),
        ):
            difference = (final_leaf - initial_leaf).reshape((num_delete, -1))
            squared_distance = squared_distance + jnp.sum(difference**2, axis=1)

        return new_particles, ReplacementUpdateInfo(
            is_accepted=move_info.is_accepted,
            num_expansions=move_info.num_expansions,
            num_shrink=move_info.num_shrink,
            bracket_left=move_info.bracket_left,
            bracket_right=move_info.bracket_right,
            has_invalid_loglikelihood=move_info.has_invalid_loglikelihood,
            start_loglikelihood=move_info.start_loglikelihood,
            contour_loglikelihood=move_info.contour_loglikelihood,
            parent_index=start_idx,
            survivor_count=survivor_count,
            start_to_final_distance=jnp.sqrt(squared_distance),
            start_chain_loglikelihood=start_state.loglikelihood,
            final_chain_loglikelihood=new_particles.loglikelihood,
        )

    return update_function


def build_prior_chord_algorithm(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    *,
    num_inner_steps: int,
    num_delete: int,
    max_shrinkage: int,
    replacement_chunks: int,
):
    """Build nested slice sampling for a uniform base measure on the unit cube.

    ``logprior_fn`` marks prior support through its finiteness; any nonuniform
    prior density must be accounted for before passing the likelihood, for
    example by transforming unit-cube coordinates in ``loglikelihood_fn``.
    Inspect ``info.update_info`` after each step to reject an invalid likelihood
    or a contour with no strict survivors before continuing the run.
    """
    init_state_fn = partial(
        init_state_strategy, logprior_fn=logprior_fn, loglikelihood_fn=loglikelihood_fn
    )

    def constrained_step(rng_key, state, loglikelihood_0, cov):
        direction_key, slice_key = random.split(rng_key)
        direction = nss.sample_direction_from_covariance(
            direction_key, state.position, cov
        )
        return prior_chord_slice(
            slice_key,
            state,
            init_state_fn,
            direction,
            loglikelihood_0,
            max_shrinkage,
        )

    step_fn = from_mcmc.build_kernel(
        constrained_step,
        num_inner_steps,
        nss.live_covariance,
        num_delete,
        update_strategy=partial(
            replacement_update_with_mcmc_take_last,
            replacement_chunks=replacement_chunks,
        ),
    )

    def init_fn(position, rng_key=None):
        return adaptive.init(
            position,
            init_state_fn=jax.vmap(init_state_fn),
            update_inner_kernel_params_fn=nss.live_covariance,
            rng_key=rng_key,
        )

    return SamplingAlgorithm(init_fn, step_fn)
