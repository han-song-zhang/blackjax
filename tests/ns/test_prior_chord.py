"""Regression tests for exact unit-cube nested slice sampling."""

import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from blackjax.ns import base, prior_chord


def _unit_prior(position):
    return jnp.where(jnp.all((position >= 0) & (position <= 1)), 0.0, -jnp.inf)


def _coordinate_likelihood(position):
    return position[0]


class TestPriorChord(parameterized.TestCase):
    def test_chord_intersects_both_cube_faces_with_a_zero_direction_component(self):
        position = jnp.array([0.25, 0.75, 0.0])
        direction = jnp.array([1.0, -2.0, 0.0])
        left, right = prior_chord.unit_chord_bounds(position, direction)
        chex.assert_trees_all_equal(left, jnp.asarray(-0.125))
        chex.assert_trees_all_equal(right, jnp.asarray(0.375))
        chex.assert_trees_all_equal(
            position + left * direction, jnp.array([0.125, 1.0, 0.0])
        )
        chex.assert_trees_all_equal(
            position + right * direction, jnp.array([0.625, 0.0, 0.0])
        )

    @parameterized.named_parameters(
        ("above", 0.0, 1.0, True, False),
        ("tie", 0.0, 0.25, False, False),
        ("outside_prior", -float("inf"), 1.0, False, False),
        ("zero_likelihood", 0.0, -float("inf"), False, False),
        ("positive_infinity", 0.0, float("inf"), False, True),
        ("nan", 0.0, float("nan"), False, True),
    )
    def test_prior_and_likelihood_support(
        self, logdensity, loglikelihood, accepted, invalid
    ):
        contour = jnp.asarray(0.25)
        initial = base.StateWithLogLikelihood(
            jnp.asarray([0.5]),
            jnp.asarray(0.0),
            jnp.asarray(0.5),
            jnp.asarray(-jnp.inf),
        )

        def init(position, loglikelihood_birth):
            return base.StateWithLogLikelihood(
                position,
                jnp.asarray(logdensity),
                jnp.asarray(loglikelihood),
                loglikelihood_birth,
            )

        def move(key, state):
            return prior_chord.prior_chord_slice(
                key, state, init, jnp.ones_like(state.position), contour, 3
            )

        new, info = jax.jit(move)(jax.random.key(0), initial)
        self.assertEqual(bool(info.is_accepted), accepted)
        self.assertEqual(bool(info.has_invalid_loglikelihood), invalid)
        self.assertEqual(int(info.num_shrink), 1 if accepted else 3)
        chex.assert_trees_all_equal(new.loglikelihood_birth, contour)
        if accepted:
            chex.assert_trees_all_equal(new.loglikelihood, jnp.asarray(loglikelihood))
            self.assertTrue(bool(jnp.all((new.position >= 0) & (new.position <= 1))))
        else:
            chex.assert_trees_all_equal(new.position, initial.position)
            chex.assert_trees_all_equal(new.loglikelihood, initial.loglikelihood)

    def test_chunking_preserves_keys_survivors_and_order(self):
        values = jnp.asarray([0.0, 0.25, 0.25, 0.125, 0.375, 0.5, 0.75, 1.0])
        live = base.NSState(
            base.StateWithLogLikelihood(
                values[:, None],
                jnp.zeros_like(values),
                values,
                jnp.full_like(values, -jnp.inf),
            )
        )

        def init(position, loglikelihood_birth):
            return base.init_state_strategy(
                position, _unit_prior, _coordinate_likelihood, loglikelihood_birth
            )

        def move(key, state, loglikelihood_0):
            return prior_chord.prior_chord_slice(
                key, state, init, jnp.ones_like(state.position), loglikelihood_0, 3
            )

        def update(chunks, contour):
            replace = prior_chord.replacement_update_with_mcmc_take_last(
                move, 3, 4, chunks
            )
            return jax.jit(replace)(jax.random.key(1919), live, jnp.asarray(contour))

        particles, info = update(1, 0.25)
        for chunks in (2, 4):
            other_particles, other_info = update(chunks, 0.25)
            chex.assert_trees_all_equal(particles, other_particles)
            chex.assert_trees_all_equal(info, other_info)

        self.assertEqual(int(info.survivor_count), 4)
        self.assertTrue(bool(jnp.all(values[info.parent_index] > 0.25)))
        self.assertTrue(bool(jnp.all(particles.loglikelihood > 0.25)))
        self.assertTrue(bool(jnp.any(info.is_accepted)))
        chex.assert_trees_all_equal(
            info.start_chain_loglikelihood, values[info.parent_index]
        )
        chex.assert_trees_all_equal(
            info.final_chain_loglikelihood, particles.loglikelihood
        )
        chex.assert_trees_all_equal(particles.loglikelihood_birth, jnp.full((4,), 0.25))

        _, empty_info = update(2, 1.0)
        self.assertEqual(int(empty_info.survivor_count), 0)
        self.assertFalse(bool(jnp.any(empty_info.is_accepted)))

    def test_all_rejected_replacements_are_born_at_current_contour(self):
        algorithm = prior_chord.build_prior_chord_algorithm(
            _unit_prior,
            _coordinate_likelihood,
            num_inner_steps=2,
            num_delete=4,
            max_shrinkage=0,
            replacement_chunks=2,
        )
        values = jnp.arange(1, 9, dtype=float) / 8.0
        initial = algorithm.init(values[:, None], jax.random.key(0))
        state, dead = jax.jit(algorithm.step)(jax.random.key(1), initial)
        info = dead.update_info
        self.assertFalse(bool(jnp.any(info.is_accepted)))
        chex.assert_trees_all_equal(
            state.particles.position[:4], values[info.parent_index, None]
        )
        chex.assert_trees_all_equal(
            state.particles.loglikelihood_birth[:4], jnp.full((4,), 0.5)
        )


if __name__ == "__main__":
    absltest.main()
