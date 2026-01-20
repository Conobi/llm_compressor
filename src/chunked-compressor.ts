/**
 * Chunked LLM compressor with parallel decompression support.
 *
 * Splits text into independent chunks that can be decompressed in parallel,
 * trading a small amount of compression ratio for significantly faster
 * decompression on multi-core systems.
 */

import { BitOutputStream, BitInputStream } from './core/bit-stream.js';
import { ArithmeticEncoder } from './core/arithmetic-encoder.js';
import { ArithmeticDecoder } from './core/arithmetic-decoder.js';
import { BPETokenizer } from './tokenizer/bpe-tokenizer.js';
import { RWKVSession, type RWKVSessionOptions } from './model/rwkv-session.js';
import {
  type ChunkedHeader,
  type ChunkConfig,
  DEFAULT_CHUNK_CONFIG,
  createChunkedHeader,
  serializeChunkedHeader,
  deserializeChunkedHeader,
  calculateChunkedHeaderSize,
  splitIntoChunks,
  isChunkedFormat,
} from './format/chunked-header.js';
import type { ProgressInfo } from './compressor.js';

/**
 * Options for ChunkedCompressor initialization.
 */
export interface ChunkedCompressorOptions {
  /** Path/URL to ONNX model file, or ArrayBuffer of model data */
  model: string | ArrayBuffer;

  /** Path/URL to tokenizer.json, or parsed tokenizer config string */
  tokenizer: string;

  /** Number of WASM threads (default: auto-detect) */
  wasmThreads?: number;

  /** Chunk configuration */
  chunkConfig?: Partial<ChunkConfig>;

  /** Progress callback */
  onProgress?: (progress: ProgressInfo) => void;
}

/**
 * Result of chunked compression operation.
 */
export interface ChunkedCompressionResult {
  /** Compressed data (header + chunk payloads) */
  data: Uint8Array;

  /** Original text size in bytes (UTF-8) */
  originalSize: number;

  /** Compressed size in bytes */
  compressedSize: number;

  /** Compression ratio (originalSize / compressedSize) */
  compressionRatio: number;

  /** Number of tokens in the original text */
  tokenCount: number;

  /** Number of chunks */
  chunkCount: number;

  /** Tokens per chunk (configured) */
  chunkSize: number;

  /** Overlap tokens (configured) */
  overlapSize: number;
}

/**
 * LLM-based text compressor with chunked parallel decompression.
 *
 * Usage:
 * ```typescript
 * const compressor = new ChunkedCompressor({
 *   model: './rwkv-4-pile-169m-uint8.onnx',
 *   tokenizer: './20B_tokenizer.json',
 *   chunkConfig: { chunkSize: 128, overlapSize: 16 },
 * });
 *
 * await compressor.init();
 *
 * const result = await compressor.compress('Hello, world!');
 * const text = await compressor.decompress(result.data); // Parallel!
 *
 * compressor.dispose();
 * ```
 */
export class ChunkedCompressor {
  private options: ChunkedCompressorOptions;
  private chunkConfig: ChunkConfig;
  private model: RWKVSession | null = null;
  private tokenizer: BPETokenizer | null = null;
  private initialized: boolean = false;

  constructor(options: ChunkedCompressorOptions) {
    this.options = options;
    this.chunkConfig = {
      ...DEFAULT_CHUNK_CONFIG,
      ...options.chunkConfig,
    };
  }

  /**
   * Initialize the compressor by loading model and tokenizer.
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
   * Compress text using chunked encoding.
   */
  async compress(text: string): Promise<ChunkedCompressionResult> {
    this.ensureInitialized();

    const originalBytes = new TextEncoder().encode(text);
    const originalSize = originalBytes.length;

    // Tokenize
    this.reportProgress('tokenizing', 0, 1);
    const tokens = this.tokenizer!.encode(text);
    const tokenCount = tokens.length;

    if (tokenCount === 0) {
      const header = createChunkedHeader(
        0,
        0,
        this.model!.getModelHash(),
        this.chunkConfig.chunkSize,
        this.chunkConfig.overlapSize,
        [],
        []
      );
      const headerBytes = serializeChunkedHeader(header);
      return {
        data: headerBytes,
        originalSize: 0,
        compressedSize: headerBytes.length,
        compressionRatio: 1,
        tokenCount: 0,
        chunkCount: 0,
        chunkSize: this.chunkConfig.chunkSize,
        overlapSize: this.chunkConfig.overlapSize,
      };
    }

    // Split into chunks
    const chunks = splitIntoChunks(
      tokens,
      this.chunkConfig.chunkSize,
      this.chunkConfig.overlapSize
    );

    // Compress each chunk sequentially (model is stateful)
    const compressedChunks: Uint8Array[] = [];
    const chunkTokenCounts: number[] = [];

    for (let i = 0; i < chunks.length; i++) {
      this.reportProgress('compressing', i, chunks.length);

      const chunk = chunks[i];
      const compressedChunk = await this.compressChunk(
        chunk.tokens,
        i === 0 ? 0 : this.chunkConfig.overlapSize
      );

      compressedChunks.push(compressedChunk);
      // Store full token count (including overlap) for decompression
      chunkTokenCounts.push(chunk.tokens.length);
    }

    // Calculate chunk offsets
    const headerSize = calculateChunkedHeaderSize(chunks.length);
    const chunkOffsets: number[] = [];
    let currentOffset = headerSize;

    for (const chunk of compressedChunks) {
      chunkOffsets.push(currentOffset);
      currentOffset += chunk.length;
    }

    // Create header
    const header = createChunkedHeader(
      originalSize,
      tokenCount,
      this.model!.getModelHash(),
      this.chunkConfig.chunkSize,
      this.chunkConfig.overlapSize,
      chunkOffsets,
      chunkTokenCounts
    );

    // Combine header and chunks
    const headerBytes = serializeChunkedHeader(header);
    const totalSize =
      headerBytes.length +
      compressedChunks.reduce((sum, c) => sum + c.length, 0);
    const data = new Uint8Array(totalSize);

    data.set(headerBytes, 0);
    let offset = headerBytes.length;
    for (const chunk of compressedChunks) {
      data.set(chunk, offset);
      offset += chunk.length;
    }

    return {
      data,
      originalSize,
      compressedSize: data.length,
      compressionRatio: originalSize / data.length,
      tokenCount,
      chunkCount: chunks.length,
      chunkSize: this.chunkConfig.chunkSize,
      overlapSize: this.chunkConfig.overlapSize,
    };
  }

  /**
   * Decompress chunked data with parallel chunk processing.
   */
  async decompress(data: Uint8Array, parallel: boolean = true): Promise<string> {
    this.ensureInitialized();

    if (!isChunkedFormat(data)) {
      throw new Error('Data is not in chunked format');
    }

    const header = deserializeChunkedHeader(data);

    // Validate model hash
    const expectedHash = this.model!.getModelHash();
    if (header.modelHash !== expectedHash) {
      console.warn(
        `Model hash mismatch: expected ${expectedHash}, got ${header.modelHash}. ` +
          'Decompression may fail or produce incorrect results.'
      );
    }

    if (header.chunkCount === 0) {
      return '';
    }

    // Extract chunk payloads
    const chunkPayloads: Uint8Array[] = [];
    for (let i = 0; i < header.chunkCount; i++) {
      const start = header.chunkOffsets[i];
      const end =
        i < header.chunkCount - 1
          ? header.chunkOffsets[i + 1]
          : data.length;
      chunkPayloads.push(data.slice(start, end));
    }

    let allTokens: number[];

    if (parallel && header.chunkCount > 1) {
      // Parallel decompression
      allTokens = await this.decompressChunksParallel(header, chunkPayloads);
    } else {
      // Sequential decompression
      allTokens = await this.decompressChunksSequential(header, chunkPayloads);
    }

    // Detokenize
    this.reportProgress('detokenizing', 0, 1);
    return this.tokenizer!.decode(allTokens);
  }

  /**
   * Decompress chunks in parallel using multiple model instances.
   */
  private async decompressChunksParallel(
    header: ChunkedHeader,
    chunkPayloads: Uint8Array[]
  ): Promise<number[]> {
    // Create additional model instances for parallel processing
    const modelInstances: RWKVSession[] = [this.model!];

    // Create N-1 additional model instances (reuse the main one)
    const additionalCount = Math.min(header.chunkCount - 1, 3); // Cap at 4 total
    for (let i = 0; i < additionalCount; i++) {
      const instance = new RWKVSession({
        model: this.options.model,
        wasmThreads: this.options.wasmThreads,
      });
      await instance.init();
      modelInstances.push(instance);
    }

    try {
      // Assign chunks to model instances
      const chunkPromises: Promise<number[]>[] = [];

      for (let i = 0; i < header.chunkCount; i++) {
        const modelIndex = i % modelInstances.length;
        const model = modelInstances[modelIndex];
        // Token count is stored directly (includes overlap)
        const tokenCount = header.chunkTokenCounts[i];

        // Each chunk decompression is independent
        chunkPromises.push(
          this.decompressChunkWithModel(
            model,
            chunkPayloads[i],
            tokenCount,
            i === 0 ? 0 : header.overlapSize
          ).then((tokens) => {
            this.reportProgress('decompressing', i, header.chunkCount);
            return tokens;
          })
        );
      }

      // Wait for all chunks (they run with some parallelism due to async)
      const chunkResults = await Promise.all(chunkPromises);

      // Merge results (skip overlap tokens except for first chunk)
      const allTokens: number[] = [];
      for (let i = 0; i < chunkResults.length; i++) {
        const tokens = chunkResults[i];
        const skipCount = i === 0 ? 0 : header.overlapSize;
        allTokens.push(...tokens.slice(skipCount));
      }

      return allTokens;
    } finally {
      // Dispose additional model instances (keep the main one)
      for (let i = 1; i < modelInstances.length; i++) {
        modelInstances[i].dispose();
      }
    }
  }

  /**
   * Decompress chunks sequentially using the main model instance.
   */
  private async decompressChunksSequential(
    header: ChunkedHeader,
    chunkPayloads: Uint8Array[]
  ): Promise<number[]> {
    const allTokens: number[] = [];

    for (let i = 0; i < header.chunkCount; i++) {
      this.reportProgress('decompressing', i, header.chunkCount);

      // Token count is stored directly (includes overlap)
      const tokenCount = header.chunkTokenCounts[i];

      const tokens = await this.decompressChunkWithModel(
        this.model!,
        chunkPayloads[i],
        tokenCount,
        i === 0 ? 0 : header.overlapSize
      );

      // Skip overlap tokens except for first chunk
      const skipCount = i === 0 ? 0 : header.overlapSize;
      allTokens.push(...tokens.slice(skipCount));
    }

    return allTokens;
  }

  /**
   * Compress a single chunk of tokens.
   *
   * For parallel decompression, we encode ALL tokens including overlap.
   * This means overlap tokens are encoded redundantly across chunks,
   * which is the trade-off for enabling parallel decompression.
   */
  private async compressChunk(
    tokens: number[],
    _contextTokens: number // Kept for API compatibility but not used
  ): Promise<Uint8Array> {
    // Reset model state for this chunk
    this.model!.reset();

    const bitStream = new BitOutputStream();
    const encoder = new ArithmeticEncoder(bitStream);

    // Encode ALL tokens (including overlap for parallel decompression)
    for (let i = 0; i < tokens.length; i++) {
      const contextToken = i === 0 ? 0 : tokens[i - 1];
      const probs = await this.model!.processToken(contextToken);
      encoder.encode(tokens[i], probs);
    }

    encoder.finish();
    return bitStream.toUint8Array();
  }

  /**
   * Decompress a single chunk with a specific model instance.
   *
   * Each chunk is fully self-contained (includes overlap tokens),
   * enabling independent parallel decompression.
   */
  private async decompressChunkWithModel(
    model: RWKVSession,
    payload: Uint8Array,
    totalTokens: number,
    _contextTokens: number // Kept for API compatibility
  ): Promise<number[]> {
    model.reset();

    const bitStream = new BitInputStream(payload);
    const decoder = new ArithmeticDecoder(bitStream);
    const tokens: number[] = [];

    // Decode all tokens (all tokens are encoded, including overlap)
    for (let i = 0; i < totalTokens; i++) {
      const contextToken = i === 0 ? 0 : tokens[i - 1];
      const probs = await model.processToken(contextToken);
      const token = decoder.decode(probs);
      tokens.push(token);
    }

    return tokens;
  }

  /**
   * Get the current chunk configuration.
   */
  getChunkConfig(): ChunkConfig {
    return { ...this.chunkConfig };
  }

  /**
   * Release resources.
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

  private ensureInitialized(): void {
    if (!this.initialized) {
      throw new Error('Compressor not initialized. Call init() first.');
    }
  }

  private reportProgress(
    stage: ProgressInfo['stage'],
    current: number,
    total: number
  ): void {
    this.options.onProgress?.({ stage, current, total });
  }
}
