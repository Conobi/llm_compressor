/**
 * RWKV model state management.
 *
 * RWKV-4 maintains 5 state tensors that persist between token generations.
 * These must be properly initialized and updated for correct model operation.
 */

/**
 * RWKV-4 model state structure.
 */
export interface RWKVState {
  /** Attention x state: [n_layer, n_embd] */
  xx_att: Float32Array;

  /** Attention a state: [n_layer, n_embd] */
  aa_att: Float32Array;

  /** Attention b state: [n_layer, n_embd] */
  bb_att: Float32Array;

  /** Attention p state (initialized to -1e30): [n_layer, n_embd] */
  pp_att: Float32Array;

  /** FFN x state: [n_layer, n_embd] */
  xx_ffn: Float32Array;
}

/**
 * RWKV model configuration.
 */
export interface RWKVConfig {
  n_layer: number;
  n_embd: number;
  n_vocab: number;
  ctx_len: number;
}

/**
 * Configuration for RWKV-4-pile-169m model.
 */
export const RWKV_169M_CONFIG: RWKVConfig = {
  n_layer: 12,
  n_embd: 768,
  n_vocab: 50277, // GPT-NeoX tokenizer vocabulary size
  ctx_len: 1024,
};

/**
 * Create initial state for RWKV model.
 *
 * @param config - Model configuration
 * @returns Initialized state with proper dimensions
 */
export function createInitialState(config: RWKVConfig): RWKVState {
  const size = config.n_layer * config.n_embd;

  // pp_att must be initialized to -1e30 (negative infinity for softmax)
  const pp_att = new Float32Array(size);
  pp_att.fill(-1e30);

  return {
    xx_att: new Float32Array(size),
    aa_att: new Float32Array(size),
    bb_att: new Float32Array(size),
    pp_att,
    xx_ffn: new Float32Array(size),
  };
}

/**
 * Clone a state object (deep copy).
 *
 * @param state - State to clone
 * @returns New state with copied data
 */
export function cloneState(state: RWKVState): RWKVState {
  return {
    xx_att: new Float32Array(state.xx_att),
    aa_att: new Float32Array(state.aa_att),
    bb_att: new Float32Array(state.bb_att),
    pp_att: new Float32Array(state.pp_att),
    xx_ffn: new Float32Array(state.xx_ffn),
  };
}

/**
 * Copy state data from source to target (in-place).
 *
 * @param source - State to copy from
 * @param target - State to copy to
 */
export function copyState(source: RWKVState, target: RWKVState): void {
  target.xx_att.set(source.xx_att);
  target.aa_att.set(source.aa_att);
  target.bb_att.set(source.bb_att);
  target.pp_att.set(source.pp_att);
  target.xx_ffn.set(source.xx_ffn);
}
