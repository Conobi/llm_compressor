import type * as OrtWeb from 'onnxruntime-web';
import { softmax } from './softmax.js';
import {
  type RWKVState,
  type RWKVConfig,
  RWKV_169M_CONFIG,
  createInitialState,
} from './rwkv-state.js';
import { detectPlatform, isCrossOriginIsolated } from '../utils/platform.js';
import { getOptimalBackend, type OnnxBackend } from './onnx-backend.js';

/**
 * Options for RWKV session initialization.
 */
export interface RWKVSessionOptions {
  /** Path/URL to ONNX model file, or ArrayBuffer of model data */
  model: string | ArrayBuffer;

  /** Number of WASM threads (default: auto-detect) */
  wasmThreads?: number;

  /** Model configuration (default: RWKV_169M_CONFIG) */
  config?: RWKVConfig;
}

/**
 * RWKV model session wrapper using ONNX Runtime.
 *
 * This class manages:
 * - ONNX model loading and inference
 * - State tensor management between inference calls
 * - Probability distribution computation from logits
 */
export class RWKVSession {
  private session: OrtWeb.InferenceSession | null = null;
  private state: RWKVState;
  private config: RWKVConfig;
  private options: RWKVSessionOptions;
  private initialized: boolean = false;
  private ort: typeof OrtWeb | null = null;
  private backendName: string = '';

  constructor(options: RWKVSessionOptions) {
    this.options = options;
    this.config = options.config ?? RWKV_169M_CONFIG;
    this.state = createInitialState(this.config);
  }

  /**
   * Initialize the ONNX session and load the model.
   */
  async init(): Promise<void> {
    if (this.initialized) return;

    const platform = detectPlatform();

    // Get optimal backend for current environment
    const backend: OnnxBackend = await getOptimalBackend();
    this.ort = backend.ort;
    this.backendName = backend.backendName;

    console.log(`Using ONNX backend: ${backend.backendName}`);

    // Configure WASM backend settings (applies when WASM is used)
    if (backend.executionProviders.includes('wasm')) {
      this.ort.env.wasm.proxy = platform.isBrowser; // Use web worker in browser

      if (platform.supportsSharedArrayBuffer && isCrossOriginIsolated()) {
        this.ort.env.wasm.numThreads = Math.min(
          this.options.wasmThreads ?? 4,
          Math.max(1, Math.floor(platform.availableThreads / 2))
        );
      } else {
        this.ort.env.wasm.numThreads = 1;
        if (platform.isBrowser) {
          console.warn(
            'SharedArrayBuffer not available. Enable cross-origin isolation for better performance.'
          );
        }
      }
    }

    // Session options with detected execution providers
    const sessionOptions: OrtWeb.InferenceSession.SessionOptions = {
      executionProviders: backend.executionProviders,
      graphOptimizationLevel: 'all',
    };

    // Load model data
    let modelData: ArrayBufferLike;

    if (this.options.model instanceof ArrayBuffer) {
      modelData = this.options.model;
    } else if (typeof this.options.model === 'string') {
      const modelPath = this.options.model;
      const isUrl = modelPath.startsWith('http://') || modelPath.startsWith('https://');

      if (platform.isBrowser) {
        // Browser: always use fetch (works with URLs and relative paths)
        const response = await fetch(modelPath);
        if (!response.ok) {
          throw new Error(`Failed to fetch model: ${response.statusText}`);
        }
        modelData = await response.arrayBuffer();
      } else if (isUrl) {
        // Node.js with URL: use fetch
        const response = await fetch(modelPath);
        if (!response.ok) {
          throw new Error(`Failed to fetch model: ${response.statusText}`);
        }
        modelData = await response.arrayBuffer();
      } else {
        // Node.js with file path: use fs.readFile
        const fs = await import('fs/promises');
        const buffer = await fs.readFile(modelPath);
        modelData = buffer.buffer.slice(
          buffer.byteOffset,
          buffer.byteOffset + buffer.byteLength
        );
      }
    } else {
      throw new Error('Invalid model source');
    }

    // Create ONNX session
    this.session = await this.ort.InferenceSession.create(
      new Uint8Array(modelData),
      sessionOptions
    );

    this.initialized = true;
  }

  /**
   * Get vocabulary size.
   */
  get vocabSize(): number {
    return this.config.n_vocab;
  }

  /**
   * Reset model state to initial values.
   * Must be called before starting a new sequence.
   */
  reset(): void {
    this.state = createInitialState(this.config);
  }

  /**
   * Process a single token and return probability distribution for next token.
   *
   * @param token - The token ID to process (0 to vocabSize-1)
   * @returns Float32Array of probabilities, length = vocabSize
   */
  async processToken(token: number): Promise<Float32Array> {
    if (!this.session || !this.ort) {
      throw new Error('Session not initialized. Call init() first.');
    }

    // Prepare input tensor
    // The model expects a context of ctx_len tokens, left-padded with zeros
    const idxData = new Int32Array(this.config.ctx_len);
    idxData[this.config.ctx_len - 1] = token;

    // Prepare state tensors
    const stateShape = [this.config.n_layer, this.config.n_embd];

    const feeds: OrtWeb.InferenceSession.OnnxValueMapType = {
      idx: new this.ort.Tensor('int32', idxData, [this.config.ctx_len]),
      xx_att: new this.ort.Tensor('float32', this.state.xx_att, stateShape),
      aa_att: new this.ort.Tensor('float32', this.state.aa_att, stateShape),
      bb_att: new this.ort.Tensor('float32', this.state.bb_att, stateShape),
      pp_att: new this.ort.Tensor('float32', this.state.pp_att, stateShape),
      xx_ffn: new this.ort.Tensor('float32', this.state.xx_ffn, stateShape),
    };

    // Run inference
    const results = await this.session.run(feeds);

    // Update state for next iteration
    this.state.xx_att = new Float32Array(results.xx_att_r.data as Float32Array);
    this.state.aa_att = new Float32Array(results.aa_att_r.data as Float32Array);
    this.state.bb_att = new Float32Array(results.bb_att_r.data as Float32Array);
    this.state.pp_att = new Float32Array(results.pp_att_r.data as Float32Array);
    this.state.xx_ffn = new Float32Array(results.xx_ffn_r.data as Float32Array);

    // Get logits and convert to probabilities
    const logits = results.x.data as Float32Array;
    return softmax(logits);
  }

  /**
   * Process multiple tokens and return probability distributions for each step.
   *
   * @param tokens - Array of token IDs to process
   * @returns Array of Float32Array probabilities, one per token
   */
  async processTokens(tokens: number[]): Promise<Float32Array[]> {
    const probabilities: Float32Array[] = [];

    for (const token of tokens) {
      const probs = await this.processToken(token);
      probabilities.push(probs);
    }

    return probabilities;
  }

  /**
   * Get current state (for checkpointing).
   */
  getState(): RWKVState {
    return {
      xx_att: new Float32Array(this.state.xx_att),
      aa_att: new Float32Array(this.state.aa_att),
      bb_att: new Float32Array(this.state.bb_att),
      pp_att: new Float32Array(this.state.pp_att),
      xx_ffn: new Float32Array(this.state.xx_ffn),
    };
  }

  /**
   * Restore state from checkpoint.
   */
  setState(state: RWKVState): void {
    this.state = {
      xx_att: new Float32Array(state.xx_att),
      aa_att: new Float32Array(state.aa_att),
      bb_att: new Float32Array(state.bb_att),
      pp_att: new Float32Array(state.pp_att),
      xx_ffn: new Float32Array(state.xx_ffn),
    };
  }

  /**
   * Get a hash identifying this model (for header validation).
   */
  getModelHash(): number {
    // Simple hash based on config - in production, could hash actual weights
    return (
      (this.config.n_layer * 1000000 +
        this.config.n_embd * 1000 +
        this.config.n_vocab) >>>
      0
    );
  }

  /**
   * Release resources.
   */
  dispose(): void {
    this.session?.release();
    this.session = null;
    this.ort = null;
    this.initialized = false;
  }

  /**
   * Get the name of the backend being used.
   */
  getBackendName(): string {
    return this.backendName;
  }

  /**
   * Check if session is initialized.
   */
  isInitialized(): boolean {
    return this.initialized;
  }
}
