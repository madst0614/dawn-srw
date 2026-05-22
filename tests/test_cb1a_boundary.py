import jax
import jax.numpy as jnp

from models.legacy.dawn_srw_v4161 import (
    _cb1a_boundary_gaps_from_rho_tau,
    _cb1a_local_winner_gaps_from_boundaries,
)


def test_cb1a_boundary_candidates_and_gaps():
    rho = jnp.array([[[0.10, 0.40, 0.55, 0.80]]], dtype=jnp.float32)
    tau = jnp.array([[[0.50]]], dtype=jnp.float32)

    below, above, challenge, prune, valid, has_above, has_below = (
        _cb1a_boundary_gaps_from_rho_tau(rho, tau))

    assert below.shape == (1, 1, 1)
    assert above.shape == (1, 1, 1)
    assert jnp.allclose(below, 0.40)
    assert jnp.allclose(above, 0.55)
    assert jnp.allclose(challenge, 0.15)
    assert jnp.allclose(prune, 0.05)
    assert bool(valid[0, 0, 0])
    assert bool(has_above[0, 0, 0])
    assert bool(has_below[0, 0, 0])


def test_cb1a_challenge_gradient_hits_below_not_tau_or_anchor():
    tau = jnp.array([[[0.50]]], dtype=jnp.float32)

    def loss_rho(rho):
        _, _, challenge, _, _, _, _ = _cb1a_boundary_gaps_from_rho_tau(rho, tau)
        return challenge.sum()

    rho = jnp.array([[[0.10, 0.40, 0.55, 0.80]]], dtype=jnp.float32)
    grad_rho = jax.grad(loss_rho)(rho)

    assert jnp.allclose(grad_rho[0, 0, 1], -1.0)
    assert jnp.allclose(grad_rho[0, 0, 2], 0.0)

    def loss_tau(tau_value):
        _, _, challenge, _, _, _, _ = _cb1a_boundary_gaps_from_rho_tau(rho, tau_value)
        return challenge.sum()

    assert jnp.allclose(jax.grad(loss_tau)(tau), 0.0)


def test_cb1a_prune_gradient_hits_above_not_tau():
    rho = jnp.array([[[0.10, 0.40, 0.55, 0.80]]], dtype=jnp.float32)

    def loss_rho(rho_value):
        _, _, _, prune, _, _, _ = _cb1a_boundary_gaps_from_rho_tau(
            rho_value, jnp.array([[[0.50]]], dtype=jnp.float32))
        return prune.sum()

    grad_rho = jax.grad(loss_rho)(rho)

    assert jnp.allclose(grad_rho[0, 0, 2], 1.0)
    assert jnp.allclose(grad_rho[0, 0, 1], 0.0)

    def loss_tau(tau_value):
        _, _, _, prune, _, _, _ = _cb1a_boundary_gaps_from_rho_tau(rho, tau_value)
        return prune.sum()

    assert jnp.allclose(
        jax.grad(loss_tau)(jnp.array([[[0.50]]], dtype=jnp.float32)), 0.0)


def test_cb1a_local_winner_gradient_avoids_global_refs():
    local_below = jnp.array([[[0.40]]], dtype=jnp.float32)
    local_above = jnp.array([[[0.55]]], dtype=jnp.float32)
    tau = jnp.array([[[0.50]]], dtype=jnp.float32)
    global_below = jnp.array([[[0.40]]], dtype=jnp.float32)
    global_above = jnp.array([[[0.55]]], dtype=jnp.float32)

    def challenge_loss(local_below_value, global_above_value):
        challenge, _, _, _, _, _, _ = _cb1a_local_winner_gaps_from_boundaries(
            local_below_value, local_above, tau, global_below, global_above_value)
        return challenge.sum()

    grad_local_below, grad_global_above = jax.grad(
        challenge_loss, argnums=(0, 1))(local_below, global_above)

    assert jnp.allclose(grad_local_below, -1.0)
    assert jnp.allclose(grad_global_above, 0.0)

    def prune_loss(local_above_value, tau_value):
        _, prune, _, _, _, _, _ = _cb1a_local_winner_gaps_from_boundaries(
            local_below, local_above_value, tau_value, global_below, global_above)
        return prune.sum()

    grad_local_above, grad_tau = jax.grad(
        prune_loss, argnums=(0, 1))(local_above, tau)

    assert jnp.allclose(grad_local_above, 1.0)
    assert jnp.allclose(grad_tau, 0.0)


def test_cb1a_non_winner_has_no_local_gradient():
    local_below = jnp.array([[[0.35]]], dtype=jnp.float32)
    local_above = jnp.array([[[0.60]]], dtype=jnp.float32)
    tau = jnp.array([[[0.50]]], dtype=jnp.float32)
    global_below = jnp.array([[[0.40]]], dtype=jnp.float32)
    global_above = jnp.array([[[0.55]]], dtype=jnp.float32)

    def loss(local_below_value, local_above_value):
        challenge, prune, _, _, _, _, _ = _cb1a_local_winner_gaps_from_boundaries(
            local_below_value, local_above_value, tau, global_below, global_above)
        return challenge.sum() + prune.sum()

    grad_below, grad_above = jax.grad(loss, argnums=(0, 1))(
        local_below, local_above)

    assert jnp.allclose(grad_below, 0.0)
    assert jnp.allclose(grad_above, 0.0)
