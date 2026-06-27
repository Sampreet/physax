import jax
import jax.numpy as jnp
from physax.virtual_machine import VirtualMachine

def run_batch_until_division(agents, max_steps, keys, cfg):
    vm = VirtualMachine(cfg)
    
    def cond_fun(val):
        a_batch, step, keys, finished_steps = val
        all_divided = jnp.all(a_batch.has_child)
        within_limits = step < max_steps
        return ~all_divided & within_limits

    def body_fun(val):
        a_batch, step, keys, finished_steps = val
        new_a_batch = jax.vmap(vm.execute_one)(a_batch, keys)
        keys = jax.vmap(jax.random.split)(keys)[:, 1]
        
        just_finished = new_a_batch.has_child & ~a_batch.has_child
        new_finished_steps = jnp.where(just_finished, step + 1, finished_steps)
        
        return new_a_batch, step + 1, keys, new_finished_steps

    init_finished = jnp.full(agents.age.shape[0], -1, dtype=jnp.int32)
    final_agents, final_steps, _, final_finished_steps = jax.lax.while_loop(
        cond_fun, body_fun, (agents, jnp.int32(0), keys, init_finished)
    )
    return final_agents, final_steps, keys, final_finished_steps
