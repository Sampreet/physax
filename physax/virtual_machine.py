import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
from physax.config import Config, N_OPERANDS, BLANK, NOP, UP_IS_SIZE, OpState, OpArgs, get_opcode_functions, tape_read, UNCLASSIFIED
from physax.agent import Agent


class VirtualMachine:
    """Virtual Machine for executing organism genomes using an elegant opcode dispatch table."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.opcode_functions = get_opcode_functions(cfg)

    def update(self, agent: Agent, key) -> Agent:
        """Execute steps_per_update compound instructions on one organism."""
        def step_fn(current_agent: Agent, step_key):
            should_exec = current_agent.can_execute
            new_agent = self.execute_one(current_agent, step_key)
            # Only apply if should execute
            result = jax.tree.map(
                lambda n, o: jnp.where(should_exec, n, o),
                new_agent, current_agent
            )
            return result, None

        keys = random.split(key, self.cfg.steps_per_update)
        updated_agent, _ = lax.scan(step_fn, agent, keys)
        return updated_agent

    def tape_fetch_inst(self, genome, genome_len, ip_val):
        """Fetch instruction from parent memory only."""
        in_bounds = (ip_val >= 0) & (ip_val < genome_len)
        idx = jnp.clip(ip_val, 0, genome_len - 1)
        return jnp.where(in_bounds, genome[idx], BLANK)

    def execute_one(self, agent: Agent, key) -> Agent:
        """Execute one compound instruction."""
        ip_val = agent.se_values[0]  # SE[0] = IP

        # 1. Fetch instruction from parent tape
        fetched = self.tape_fetch_inst(agent.genome, agent.genome_len, ip_val)

        # Mark IP position as executed
        in_bounds = (ip_val >= 0) & (ip_val < agent.genome_len)
        clip_ip = jnp.clip(ip_val, 0, self.cfg.max_genome_len - 1)
        executed = jnp.where(in_bounds, agent.executed.at[clip_ip].set(True), agent.executed)

        # 2. Map to instruction index
        safe_n_instr = jnp.maximum(agent.n_instructions, 1)
        instr_idx = jnp.abs(fetched) % safe_n_instr
        instr_idx = jnp.where(agent.n_instructions > 0, instr_idx, 0)
        instr = agent.instruction_table[instr_idx]
        instr_len = agent.instruction_lengths[instr_idx]

        # 3. Execute micro-ops
        # The fillOperands logic: read operands from instruction array;
        # if not enough, read from tape and advance IP.
        # We need to track: position in instruction array, IP (for overflow reads),
        # SE values, child state, divide_returned flag, and a "next_opcode_pos"
        # to know which positions are opcodes vs operands.
        # Pre-split keys for all micro-op steps
        step_keys = random.split(key, self.cfg.max_micro_ops + 1)

        def micro_op_step(ctx_tuple, step_idx):
            ctx_state, ctx_args = ctx_tuple
            step_key = step_keys[step_idx]
            # Are we at a valid position?
            at_valid = (ctx_args.pos_in_instr < instr_len) & ~ctx_state.divide_returned

            # Read current value from instruction
            safe_pos = jnp.clip(ctx_args.pos_in_instr, 0, self.cfg.max_micro_ops - 1)
            cur_val = instr[safe_pos]

            # Is this position an opcode?
            is_opcode = (ctx_args.pos_in_instr == ctx_args.next_opcode_pos) & at_valid
            # Get opcode (already normalized during parsing)
            opcode = jnp.where(is_opcode, cur_val, NOP)
            safe_opcode = jnp.clip(opcode, 0, UP_IS_SIZE - 1)
            # Get number of operands for this opcode
            n_ops = jnp.where(is_opcode, N_OPERANDS[safe_opcode], jnp.int32(0))

            # fillOperands: read n_ops operands starting from pos_in_instr+1
            # If not enough in instruction, fetch from tape (parent) and advance ip
            ops_start = ctx_args.pos_in_instr + 1
            remaining_in_instr = jnp.maximum(instr_len - ops_start, 0)

            # Read up to 3 operands
            def read_operand(op_idx, ip_ov):
                instr_pos = ops_start + op_idx
                from_instr = op_idx < remaining_in_instr
                safe_ipos = jnp.clip(instr_pos, 0, self.cfg.max_micro_ops - 1)
                instr_val = instr[safe_ipos]
                
                # Create mini args for tape_read overflow fetch
                tape_read_args = ctx_args._replace(ip_for_overflow=ip_ov)
                tape_val = tape_read(ctx_state, tape_read_args, ip_ov)
                
                # Track if read from child tape
                total_size = jnp.maximum(ctx_args.tape_size, 1)
                read_child = (jnp.abs(ip_ov) % total_size) >= ctx_args.genome_len
                read_child_actual = ~from_instr & read_child
                
                val = jnp.where(from_instr, instr_val, tape_val)
                # Advance IP only when reading from tape
                new_ip = jnp.where(from_instr, ip_ov, ip_ov + 1)
                return val, new_ip, read_child_actual

            op0, ip_ov1, rc0 = read_operand(jnp.int32(0), ctx_args.ip_for_overflow)
            op1, ip_ov2, rc1 = read_operand(jnp.int32(1), ip_ov1)
            op2, ip_ov3, rc2 = read_operand(jnp.int32(2), ip_ov2)

            # Combine read child actual based on n_ops
            any_read_child = jnp.where(n_ops >= 3, rc0 | rc1 | rc2,
                                jnp.where(n_ops >= 2, rc0 | rc1,
                                 jnp.where(n_ops >= 1, rc0, False)))
            
            ctx_state = ctx_state._replace(read_from_child=ctx_state.read_from_child | any_read_child)

            # Select appropriate IP based on actual n_ops
            ip_after_ops = jnp.where(
                n_ops >= 3, ip_ov3, 
                jnp.where(n_ops >= 2, ip_ov2, 
                          jnp.where(n_ops >= 1, ip_ov1, ctx_args.ip_for_overflow))
            )

            # Normalize operands: abs(val) % Short.MAX_VALUE, then % n_ses for SE indexing
            def norm_op(val):
                v = jnp.where(val < 0, -val, val) % 32767
                return v % jnp.maximum(ctx_args.n_ses, 1)

            ctx_args = ctx_args._replace(
                step_key=step_key,
                o0=norm_op(op0),
                o1=norm_op(op1),
                o2=norm_op(op2)
            )

            # lax.switch evaluates only the branch matching `safe_opcode`
            # ---- Execute opcode ----
            new_state = lax.switch(safe_opcode, self.opcode_functions, (ctx_state, ctx_args))

            # If it wasn't an opcode position, we revert the state (NO-OP)
            new_state = jax.tree.map(lambda n, o: jnp.where(is_opcode, n, o), new_state, ctx_state)

            # Advance counts and positions
            ops_consumed_from_instr = jnp.minimum(n_ops, remaining_in_instr)
            new_pos = jnp.where(is_opcode, ctx_args.pos_in_instr + 1 + ops_consumed_from_instr, ctx_args.pos_in_instr)

            new_args = ctx_args._replace(
                cntr=jnp.where(is_opcode, ctx_args.cntr + 1, ctx_args.cntr),
                pos_in_instr=new_pos,
                next_opcode_pos=jnp.where(is_opcode, new_pos, ctx_args.next_opcode_pos),
                ip_for_overflow=jnp.where(is_opcode, ip_after_ops, ctx_args.ip_for_overflow)
            )

            return (new_state, new_args), None


        tape_sz = agent.genome_len + jnp.where(agent.already_allocated, agent.child_len, 0)
        
        init_state = OpState(
            se_vals=agent.se_values,
            child_arr=agent.child,
            child_cop=agent.child_copied,
            genome_arr=agent.genome,
            already_alloc=agent.already_allocated,
            child_l=agent.child_len,
            gest_time=agent.gestation_time,
            exec_inst=agent.executed_instructions,
            has_ch=agent.has_child,
            divide_returned=jnp.bool_(False),
            did_jump=jnp.bool_(False),
            read_from_child=jnp.bool_(False)
        )
        
        init_args = OpArgs(
            step_key=step_keys[0],
            genome_len=agent.genome_len,
            n_ses=agent.n_ses,
            tape_size=tape_sz,
            cntr=agent.counter,
            o0=jnp.int32(0),
            o1=jnp.int32(0),
            o2=jnp.int32(0),
            pos_in_instr=jnp.int32(0),
            next_opcode_pos=jnp.int32(0),
            ip_for_overflow=ip_val
        )

        (final_state, final_args), _ = lax.scan(micro_op_step, (init_state, init_args), jnp.arange(self.cfg.max_micro_ops))

        new_ip = final_state.se_vals[0] + 1
        new_ip = jnp.where(final_state.divide_returned, final_state.se_vals[0], new_ip)

        do_divide_success = final_state.has_ch & ~agent.has_child
        restart_ip = (agent.separator_pos + 1) % jnp.maximum(agent.genome_len, 1)
        new_ip = jnp.where(do_divide_success, restart_ip, new_ip)
        final_counter = jnp.where(do_divide_success, jnp.int32(0), final_args.cntr)

        final_se_vals = final_state.se_vals.at[0].set(new_ip)

        return agent._replace(
            se_values=final_se_vals,
            child=final_state.child_arr,
            child_len=final_state.child_l,
            child_copied=final_state.child_cop,
            already_allocated=final_state.already_alloc,
            genome=final_state.genome_arr,
            has_child=final_state.has_ch,
            counter=final_counter,
            gestation_time=final_state.gest_time,
            executed_instructions=final_state.exec_inst,
            executed=executed
        )
