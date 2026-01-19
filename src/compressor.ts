import { BitOutputStream, BitInputStream } from './core/bit-stream.js';
import { ArithmeticEncoder } from './core/arithmetic-encoder.js';
import { ArithmeticDecoder } from './core/arithmetic-decoder.js';
import { BPETokenizer } from './tokenizer/bpe-tokenizer.js';
import { RWKVSession, type RWKVSessionOptions } from './model/rwkv-session.js';
import {
  createHeader,
  serializeHeader,
  splitHeaderAndPayload,
  combineHeaderAndPayload,
} from './format/header.js';

/**
 * Progress information callback payload.
 */
export interface ProgressInfo {
  stage:
    | 'loading'
    | 'tokenizing'
    | 'compressing'
    | 'decompressing'
    | 'detokenizing';
  current: number;
  total: number;
  bytesProcessed?: number;
}

/**
 * Options for LLMCompressor initialization.
 */
export interface CompressorOptions {
  /** Path/URL to ONNX model file, or ArrayBuffer of model data */
  model: string | ArrayBuffer;

  /** Path/URL to tokenizer.json, or parsed tokenizer config string */
  tokenizer: string;

  /** Number of WASM threads (default: auto-detect) */
  wasmThreads?: number;

  /** Progress callback */
  onProgress?: (progress: ProgressInfo) => void;
}

/**
 * Result of compression operation.
 */
export interface CompressionResult {
  /** Compressed data (header + payload) */
  data: Uint8Array;

  /** Original text size in bytes (UTF-8) */
  originalSize: number;

  /** Compressed size in bytes */
  compressedSize: number;

  /** Compression ratio (originalSize / compressedSize) */
  compressionRatio: number;

  /** Number of tokens in the original text */
  tokenCount: number;
}

/**
 * LLM-based text compressor using RWKV and arithmetic coding.
 *
 * Usage:
 * ```typescript
 * const compressor = new LLMCompressor({
 *   model: './rwkv-4-pile-169m-uint8.onnx',
 *   tokenizer: './20B_tokenizer.json',
 * });
 *
 * await compressor.init();
 *
 * const result = await compressor.compress('Hello, world!');
 * const text = await compressor.decompress(result.data);
 *
 * compressor.dispose();
 * ```
 */
export class LLMCompressor {
  private options: CompressorOptions;
  private model: RWKVSession | null = null;
  private tokenizer: BPETokenizer | null = null;
  private initialized: boolean = false;

  constructor(options: CompressorOptions) {
    this.options = options;
  }

  /**
   * Initialize the compressor by loading model and tokenizer.
   * Must be called before compress() or decompress().
   */
  async init(): Promise<void> {
    if (this.initialized) return;

    this.reportProgress('loading', 0, 2);

    // Load tokenizer
    this.tokenizer = new BPETokenizer();
    await this.tokenizer.load(this.options.tokenizer);

    this.reportProgress('loading', 1, 2);

    // Load model
    const modelOptions: RWKVSessionOptions = {
      model: this.options.model,
      wasmThreads: this.options.wasmThreads,
    };

    this.model = new RWKVSession(modelOptions);
    await this.model.init();

    this.reportProgress('loading', 2, 2);
    this.initialized = true;
  }

  /**
   * Compress text to binary data.
   *
   * @param text - The text to compress
   * @returns Compression result with data and statistics
   */
  async compress(text: string): Promise<CompressionResult> {
    this.ensureInitialized();

    // Get original size
    const originalBytes = new TextEncoder().encode(text);
    const originalSize = originalBytes.length;

    // Step 1: Tokenize
    this.reportProgress('tokenizing', 0, 1);
    const tokens = this.tokenizer!.encode(text);
    const tokenCount = tokens.length;

    if (tokenCount === 0) {
      // Empty text - return minimal compressed form
      const header = createHeader(0, 0, this.model!.getModelHash());
      const headerBytes = serializeHeader(header);
      return {
        data: headerBytes,
        originalSize: 0,
        compressedSize: headerBytes.length,
        compressionRatio: 1,
        tokenCount: 0,
      };
    }

    // Step 2: Reset model state
    this.model!.reset();

    // Step 3: Arithmetic encode with LLM probabilities
    const bitStream = new BitOutputStream();
    const encoder = new ArithmeticEncoder(bitStream);

    // Process first token with initial state
    // We need to encode each token using the probability distribution
    // BEFORE seeing that token (predictive coding)

    for (let i = 0; i < tokens.length; i++) {
      this.reportProgress('compressing', i, tokens.length);

      // Get probability distribution for current position
      // For the first token, we use an initial token (0) to get initial probabilities
      // For subsequent tokens, we use the previous token
      const contextToken = i === 0 ? 0 : tokens[i - 1];
      const probs = await this.model!.processToken(contextToken);

      // Encode the actual token using these probabilities
      encoder.encode(tokens[i], probs);
    }

    // Finalize encoding
    encoder.finish();

    // Step 4: Create header and combine with compressed data
    const header = createHeader(originalSize, tokenCount, this.model!.getModelHash());
    const headerBytes = serializeHeader(header);
    const compressedPayload = bitStream.toUint8Array();
    const data = combineHeaderAndPayload(headerBytes, compressedPayload);

    return {
      data,
      originalSize,
      compressedSize: data.length,
      compressionRatio: originalSize / data.length,
      tokenCount,
    };
  }

  /**
   * Decompress binary data back to text.
   *
   * @param data - The compressed data (from compress())
   * @returns The original text
   */
  async decompress(data: Uint8Array): Promise<string> {
    this.ensureInitialized();

    // Step 1: Parse header
    const { header, payload } = splitHeaderAndPayload(data);

    // Validate model hash
    const expectedHash = this.model!.getModelHash();
    if (header.modelHash !== expectedHash) {
      console.warn(
        `Model hash mismatch: expected ${expectedHash}, got ${header.modelHash}. ` +
          'Decompression may fail or produce incorrect results.'
      );
    }

    // Handle empty text
    if (header.tokenCount === 0) {
      return '';
    }

    // Step 2: Reset model state (CRITICAL: must match encoder)
    this.model!.reset();

    // Step 3: Arithmetic decode
    const bitStream = new BitInputStream(payload);
    const decoder = new ArithmeticDecoder(bitStream);
    const tokens: number[] = [];

    for (let i = 0; i < header.tokenCount; i++) {
      this.reportProgress('decompressing', i, header.tokenCount);

      // Get probability distribution (same as encoder)
      const contextToken = i === 0 ? 0 : tokens[i - 1];
      const probs = await this.model!.processToken(contextToken);

      // Decode next token
      const token = decoder.decode(probs);
      tokens.push(token);
    }

    // Step 4: Detokenize
    this.reportProgress('detokenizing', 0, 1);
    const text = this.tokenizer!.decode(tokens);

    return text;
  }

  /**
   * Release resources (model, session, etc.).
   */
  dispose(): void {
    this.model?.dispose();
    this.model = null;
    this.tokenizer = null;
    this.initialized = false;
  }

  /**
   * Check if the compressor is initialized.
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Get the vocabulary size of the loaded model.
   */
  getVocabSize(): number {
    this.ensureInitialized();
    return this.model!.vocabSize;
  }

  /**
   * Ensure the compressor is initialized before operations.
   */
  private ensureInitialized(): void {
    if (!this.initialized) {
      throw new Error('Compressor not initialized. Call init() first.');
    }
  }

  /**
   * Report progress to the callback if provided.
   */
  private reportProgress(
    stage: ProgressInfo['stage'],
    current: number,
    total: number
  ): void {
    this.options.onProgress?.({ stage, current, total });
  }
}
