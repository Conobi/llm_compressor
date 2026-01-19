/**
 * notebox-compressor
 *
 * LLM-based lossless text compression using RWKV and arithmetic coding.
 *
 * @example
 * ```typescript
 * import { LLMCompressor } from 'notebox-compressor';
 *
 * const compressor = new LLMCompressor({
 *   model: './rwkv-4-pile-169m-uint8.onnx',
 *   tokenizer: './20B_tokenizer.json',
 * });
 *
 * await compressor.init();
 *
 * // Compress
 * const result = await compressor.compress('Hello, world!');
 * console.log(`Compression ratio: ${result.compressionRatio.toFixed(2)}x`);
 *
 * // Decompress
 * const text = await compressor.decompress(result.data);
 * console.log(text); // 'Hello, world!'
 *
 * compressor.dispose();
 * ```
 */

// Main compressor
export {
  LLMCompressor,
  type CompressorOptions,
  type CompressionResult,
  type ProgressInfo,
} from './compressor.js';

// Core arithmetic coding (for advanced usage)
export {
  ArithmeticEncoder,
  ArithmeticDecoder,
  BitOutputStream,
  BitInputStream,
  buildCumulativeDistribution,
  findSymbol,
  getSymbolRange,
  PROB_SCALE,
} from './core/index.js';

// Tokenizer (for advanced usage)
export { BPETokenizer, type TokenizerConfig } from './tokenizer/index.js';

// Model session (for advanced usage)
export {
  RWKVSession,
  type RWKVSessionOptions,
  type RWKVState,
  type RWKVConfig,
  RWKV_169M_CONFIG,
  createInitialState,
  softmax,
} from './model/index.js';

// File format (for advanced usage)
export {
  type CompressedHeader,
  MAGIC_BYTES,
  FORMAT_VERSION,
  HEADER_SIZE,
  createHeader,
  serializeHeader,
  deserializeHeader,
  combineHeaderAndPayload,
  splitHeaderAndPayload,
} from './format/index.js';

// Utilities
export { detectPlatform, type PlatformInfo } from './utils/index.js';
