"""Tests for bounded nested slice sampling."""

import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized

from blackjax.ns import prior_chord


class TestPriorChord(parameterized.TestCase):
    def test_chord_endpoints(self):
        origin = jnp.array([0.25, 0.75, 0.0])
        direction = jnp.array([1.0, -2.0, 0.0])
        left, right = prior_chord._unit_chord_bounds(origin, direction)
        chex.assert_trees_all_equal(
            origin + left * direction, jnp.array([0.125, 1.0, 0.0])
        )
        chex.assert_trees_all_equal(
            origin + right * direction, jnp.array([0.625, 0.0, 0.0])
        )

    @parameterized.parameters(0, 8)
    def test_unit_cube_replacements(self, max_shrinkage):
        def logprior(position):
            return jnp.where((position[0] >= 0) & (position[0] <= 0.875), 0.0, -jnp.inf)

        algorithm = prior_chord.as_top_level_api(
            logprior,
            lambda position: position[0],
            num_inner_steps=3,
            num_delete=4,
            max_shrinkage=max_shrinkage,
        )
        positions = jnp.arange(8, dtype=float)[:, None] / 8
        state = algorithm.init(positions, jax.random.key(0))
        state, dead = jax.jit(algorithm.step)(jax.random.key(1), state)
        replacements = state.particles.position[:4, 0]
        self.assertTrue(bool(jnp.all((replacements > 0.375) & (replacements <= 0.875))))
        chex.assert_trees_all_equal(
            state.particles.loglikelihood_birth[:4], jnp.full((4,), 0.375)
        )
        info = dead.update_info
        self.assertFalse(bool(jnp.any(info.num_expansions)))
        self.assertEqual(bool(jnp.any(info.is_accepted)), max_shrinkage > 0)
