/**
 * Tests for chunked compression.
 *
 * Compares compression ratios between sequential and chunked compression
 * across different text types and chunk configurations.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { LLMCompressor, ChunkedCompressor } from '../src/index.js';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Skip if model not available
const MODEL_PATH = process.env.RWKV_MODEL_PATH;
const TOKENIZER_PATH = path.join(__dirname, '../assets/20B_tokenizer.json');

const runRealTests = !!MODEL_PATH;

interface CompressionComparison {
  text: string;
  textName: string;
  sequential: {
    ratio: number;
    size: number;
    tokens: number;
  };
  chunked: {
    ratio: number;
    size: number;
    tokens: number;
    chunks: number;
    chunkSize: number;
    overlapSize: number;
  };
  ratioLoss: number; // Percentage loss in compression ratio
}

describe.skipIf(!runRealTests)('Chunked Compression: Ratio Comparison', () => {
  let sequentialCompressor: LLMCompressor;
  let chunkedCompressor64: ChunkedCompressor;
  let chunkedCompressor128: ChunkedCompressor;
  let chunkedCompressor256: ChunkedCompressor;
  let chunkedCompressorNoOverlap: ChunkedCompressor;

  const testTexts = {
    short: 'The quick brown fox jumps over the lazy dog.',

    medium: `In the beginning God created the heaven and the earth. And the earth was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters. And God said, Let there be light: and there was light. And God saw the light, that it was good: and God divided the light from the darkness.`,

    long: `Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Unlike traditional programming where rules are explicitly coded, machine learning algorithms identify patterns in data and make decisions with minimal human intervention.

There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. In supervised learning, the algorithm learns from labeled training data and makes predictions based on that data. Unsupervised learning involves training on unlabeled data to discover hidden patterns. Reinforcement learning trains agents to make sequences of decisions by rewarding desired behaviors.

Deep learning is a subset of machine learning that uses neural networks with many layers. These deep neural networks can automatically learn hierarchical representations of data, making them particularly effective for tasks like image recognition, natural language processing, and speech recognition.

The success of machine learning depends heavily on the quality and quantity of training data. Data preprocessing, feature engineering, and model selection are crucial steps in building effective machine learning systems. Cross-validation and hyperparameter tuning help optimize model performance while avoiding overfitting.`,

    repetitive: 'hello world. '.repeat(50),

    code: `function fibonacci(n: number): number {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}

function factorial(n: number): number {
  if (n <= 1) return 1;
  return n * factorial(n - 1);
}

function isPrime(n: number): boolean {
  if (n <= 1) return false;
  for (let i = 2; i * i <= n; i++) {
    if (n % i === 0) return false;
  }
  return true;
}

// Main execution
const fib10 = fibonacci(10);
const fact5 = factorial(5);
const prime17 = isPrime(17);
console.log(\`Fibonacci(10) = \${fib10}\`);
console.log(\`Factorial(5) = \${fact5}\`);
console.log(\`Is 17 prime? \${prime17}\`);`,
  };

  beforeAll(async () => {
    console.log('Loading compressors for ratio comparison tests...');
    console.log(`Model: ${MODEL_PATH}`);
    console.log(`Tokenizer: ${TOKENIZER_PATH}`);

    // Initialize sequential compressor
    sequentialCompressor = new LLMCompressor({
      model: MODEL_PATH!,
      tokenizer: TOKENIZER_PATH,
    });
    await sequentialCompressor.init();

    // Initialize chunked compressors with different configurations
    chunkedCompressor64 = new ChunkedCompressor({
      model: MODEL_PATH!,
      tokenizer: TOKENIZER_PATH,
      chunkConfig: { chunkSize: 64, overlapSize: 8 },
    });
    await chunkedCompressor64.init();

    chunkedCompressor128 = new ChunkedCompressor({
      model: MODEL_PATH!,
      tokenizer: TOKENIZER_PATH,
      chunkConfig: { chunkSize: 128, overlapSize: 16 },
    });
    await chunkedCompressor128.init();

    chunkedCompressor256 = new ChunkedCompressor({
      model: MODEL_PATH!,
      tokenizer: TOKENIZER_PATH,
      chunkConfig: { chunkSize: 256, overlapSize: 32 },
    });
    await chunkedCompressor256.init();

    chunkedCompressorNoOverlap = new ChunkedCompressor({
      model: MODEL_PATH!,
      tokenizer: TOKENIZER_PATH,
      chunkConfig: { chunkSize: 128, overlapSize: 0 },
    });
    await chunkedCompressorNoOverlap.init();

    console.log('All compressors loaded.\n');
  }, 120000);

  afterAll(() => {
    sequentialCompressor?.dispose();
    chunkedCompressor64?.dispose();
    chunkedCompressor128?.dispose();
    chunkedCompressor256?.dispose();
    chunkedCompressorNoOverlap?.dispose();
  });

  async function compareCompression(
    text: string,
    textName: string,
    chunkedCompressor: ChunkedCompressor
  ): Promise<CompressionComparison> {
    const seqResult = await sequentialCompressor.compress(text);
    const chunkResult = await chunkedCompressor.compress(text);
    const config = chunkedCompressor.getChunkConfig();

    const ratioLoss =
      ((seqResult.compressionRatio - chunkResult.compressionRatio) /
        seqResult.compressionRatio) *
      100;

    return {
      text,
      textName,
      sequential: {
        ratio: seqResult.compressionRatio,
        size: seqResult.compressedSize,
        tokens: seqResult.tokenCount,
      },
      chunked: {
        ratio: chunkResult.compressionRatio,
        size: chunkResult.compressedSize,
        tokens: chunkResult.tokenCount,
        chunks: chunkResult.chunkCount,
        chunkSize: config.chunkSize,
        overlapSize: config.overlapSize,
      },
      ratioLoss,
    };
  }

  it('should compress and decompress short text correctly', async () => {
    const text = testTexts.short;

    const seqResult = await sequentialCompressor.compress(text);
    const chunkResult = await chunkedCompressor128.compress(text);

    const seqDecompressed = await sequentialCompressor.decompress(seqResult.data);
    const chunkDecompressed = await chunkedCompressor128.decompress(chunkResult.data);

    expect(seqDecompressed).toBe(text);
    expect(chunkDecompressed).toBe(text);
  }, 60000);

  it('should compress and decompress medium text correctly', async () => {
    const text = testTexts.medium;

    const seqResult = await sequentialCompressor.compress(text);
    const chunkResult = await chunkedCompressor128.compress(text);

    const seqDecompressed = await sequentialCompressor.decompress(seqResult.data);
    const chunkDecompressed = await chunkedCompressor128.decompress(chunkResult.data);

    expect(seqDecompressed).toBe(text);
    expect(chunkDecompressed).toBe(text);
  }, 120000);

  it('should have acceptable ratio loss for medium text with chunk size 128', async () => {
    const comparison = await compareCompression(
      testTexts.medium,
      'medium',
      chunkedCompressor128
    );

    console.log('\n--- Medium Text Compression Comparison (chunk=128, overlap=16) ---');
    console.log(`Original size: ${new TextEncoder().encode(testTexts.medium).length} bytes`);
    console.log(`Tokens: ${comparison.sequential.tokens}`);
    console.log(`Sequential: ${comparison.sequential.size} bytes (${comparison.sequential.ratio.toFixed(2)}x)`);
    console.log(`Chunked: ${comparison.chunked.size} bytes (${comparison.chunked.ratio.toFixed(2)}x) [${comparison.chunked.chunks} chunks]`);
    console.log(`Ratio loss: ${comparison.ratioLoss.toFixed(1)}%`);

    // Accept up to 25% ratio loss for medium text
    expect(comparison.ratioLoss).toBeLessThan(25);
  }, 120000);

  it('should have acceptable ratio loss for long text with chunk size 128', async () => {
    const comparison = await compareCompression(
      testTexts.long,
      'long',
      chunkedCompressor128
    );

    console.log('\n--- Long Text Compression Comparison (chunk=128, overlap=16) ---');
    console.log(`Original size: ${new TextEncoder().encode(testTexts.long).length} bytes`);
    console.log(`Tokens: ${comparison.sequential.tokens}`);
    console.log(`Sequential: ${comparison.sequential.size} bytes (${comparison.sequential.ratio.toFixed(2)}x)`);
    console.log(`Chunked: ${comparison.chunked.size} bytes (${comparison.chunked.ratio.toFixed(2)}x) [${comparison.chunked.chunks} chunks]`);
    console.log(`Ratio loss: ${comparison.ratioLoss.toFixed(1)}%`);

    // Accept up to 20% ratio loss for long text (should be better due to more context)
    expect(comparison.ratioLoss).toBeLessThan(20);
  }, 180000);

  it('should have better ratio with larger chunks', async () => {
    const comparison64 = await compareCompression(testTexts.long, 'long', chunkedCompressor64);
    const comparison128 = await compareCompression(testTexts.long, 'long', chunkedCompressor128);
    const comparison256 = await compareCompression(testTexts.long, 'long', chunkedCompressor256);

    console.log('\n--- Chunk Size Comparison (Long Text) ---');
    console.log(`Chunk 64:  ratio loss = ${comparison64.ratioLoss.toFixed(1)}% (${comparison64.chunked.chunks} chunks)`);
    console.log(`Chunk 128: ratio loss = ${comparison128.ratioLoss.toFixed(1)}% (${comparison128.chunked.chunks} chunks)`);
    console.log(`Chunk 256: ratio loss = ${comparison256.ratioLoss.toFixed(1)}% (${comparison256.chunked.chunks} chunks)`);

    // Larger chunks should have better (lower) ratio loss
    expect(comparison256.ratioLoss).toBeLessThan(comparison64.ratioLoss);
  }, 300000);

  it('should show overlap impact on compression', async () => {
    const withOverlap = await compareCompression(testTexts.long, 'long', chunkedCompressor128);
    const withoutOverlap = await compareCompression(testTexts.long, 'long', chunkedCompressorNoOverlap);

    console.log('\n--- Overlap Comparison (Long Text, chunk=128) ---');
    console.log(`With overlap (16): ratio loss = ${withOverlap.ratioLoss.toFixed(1)}%`);
    console.log(`Without overlap:   ratio loss = ${withoutOverlap.ratioLoss.toFixed(1)}%`);
    console.log('Note: Overlap adds redundant encoding overhead but enables better chunk boundaries');

    // Both should have acceptable ratio loss (overlap trades ratio for parallelism quality)
    expect(withOverlap.ratioLoss).toBeLessThan(25);
    expect(withoutOverlap.ratioLoss).toBeLessThan(25);
  }, 240000);

  it('should handle repetitive text well', async () => {
    const comparison = await compareCompression(
      testTexts.repetitive,
      'repetitive',
      chunkedCompressor128
    );

    console.log('\n--- Repetitive Text Compression ---');
    console.log(`Original size: ${new TextEncoder().encode(testTexts.repetitive).length} bytes`);
    console.log(`Sequential: ${comparison.sequential.ratio.toFixed(2)}x`);
    console.log(`Chunked: ${comparison.chunked.ratio.toFixed(2)}x`);
    console.log(`Ratio loss: ${comparison.ratioLoss.toFixed(1)}%`);

    // Repetitive text should still compress well
    expect(comparison.chunked.ratio).toBeGreaterThan(2);
  }, 180000);

  it('should handle code well', async () => {
    const comparison = await compareCompression(
      testTexts.code,
      'code',
      chunkedCompressor128
    );

    console.log('\n--- Code Compression ---');
    console.log(`Original size: ${new TextEncoder().encode(testTexts.code).length} bytes`);
    console.log(`Sequential: ${comparison.sequential.ratio.toFixed(2)}x`);
    console.log(`Chunked: ${comparison.chunked.ratio.toFixed(2)}x`);
    console.log(`Ratio loss: ${comparison.ratioLoss.toFixed(1)}%`);

    // Both should compress the code
    expect(comparison.chunked.ratio).toBeGreaterThan(1);
  }, 180000);
});

describe.skipIf(!runRealTests)('Chunked Compression: Parallel Decompression', () => {
  let chunkedCompressor: ChunkedCompressor;

  const longText = `Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Unlike traditional programming where rules are explicitly coded, machine learning algorithms identify patterns in data and make decisions with minimal human intervention.

There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. In supervised learning, the algorithm learns from labeled training data and makes predictions based on that data. Unsupervised learning involves training on unlabeled data to discover hidden patterns. Reinforcement learning trains agents to make sequences of decisions by rewarding desired behaviors.

Deep learning is a subset of machine learning that uses neural networks with many layers. These deep neural networks can automatically learn hierarchical representations of data, making them particularly effective for tasks like image recognition, natural language processing, and speech recognition.`;

  beforeAll(async () => {
    chunkedCompressor = new ChunkedCompressor({
      model: MODEL_PATH!,
      tokenizer: TOKENIZER_PATH,
      chunkConfig: { chunkSize: 64, overlapSize: 8 },
    });
    await chunkedCompressor.init();
  }, 60000);

  afterAll(() => {
    chunkedCompressor?.dispose();
  });

  it('should decompress correctly with parallel=true', async () => {
    const result = await chunkedCompressor.compress(longText);
    const decompressed = await chunkedCompressor.decompress(result.data, true);

    expect(decompressed).toBe(longText);
  }, 120000);

  it('should decompress correctly with parallel=false', async () => {
    const result = await chunkedCompressor.compress(longText);
    const decompressed = await chunkedCompressor.decompress(result.data, false);

    expect(decompressed).toBe(longText);
  }, 120000);

  it('should produce same result with parallel and sequential decompression', async () => {
    const result = await chunkedCompressor.compress(longText);

    const parallelResult = await chunkedCompressor.decompress(result.data, true);
    const sequentialResult = await chunkedCompressor.decompress(result.data, false);

    expect(parallelResult).toBe(sequentialResult);
  }, 180000);
});

// Unit tests for chunked header format (always run)
describe('Chunked Header Format', () => {
  it('should serialize and deserialize header correctly', async () => {
    const {
      createChunkedHeader,
      serializeChunkedHeader,
      deserializeChunkedHeader,
    } = await import('../src/format/chunked-header.js');

    // Token counts include overlap: chunk 1 has 30 tokens,
    // chunks 2 and 3 have 8 overlap + 27 new = 35 tokens each
    const header = createChunkedHeader(
      1000, // originalLength
      84, // totalTokenCount (30 + 27 + 27, excluding overlap)
      0x12345678, // modelHash
      64, // chunkSize
      8, // overlapSize
      [25, 100, 175], // chunkOffsets
      [30, 35, 35] // chunkTokenCounts (includes overlap for chunks 2+)
    );

    const serialized = serializeChunkedHeader(header);
    const deserialized = deserializeChunkedHeader(serialized);

    expect(deserialized.originalLength).toBe(1000);
    expect(deserialized.totalTokenCount).toBe(84);
    expect(deserialized.modelHash).toBe(0x12345678);
    expect(deserialized.chunkCount).toBe(3);
    expect(deserialized.chunkSize).toBe(64);
    expect(deserialized.overlapSize).toBe(8);
    expect(deserialized.chunkOffsets).toEqual([25, 100, 175]);
    expect(deserialized.chunkTokenCounts).toEqual([30, 35, 35]);
  });

  it('should split tokens into chunks correctly', async () => {
    const { splitIntoChunks } = await import('../src/format/chunked-header.js');

    const tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

    // Chunk size 5, overlap 2
    const chunks = splitIntoChunks(tokens, 5, 2);

    expect(chunks.length).toBe(3);

    // First chunk: tokens 0-4 (no overlap prefix)
    expect(chunks[0].tokens).toEqual([1, 2, 3, 4, 5]);
    expect(chunks[0].outputStart).toBe(0);
    expect(chunks[0].outputCount).toBe(5);

    // Second chunk: tokens 3-9 (2 overlap + 5 new)
    expect(chunks[1].tokens).toEqual([4, 5, 6, 7, 8, 9, 10]);
    expect(chunks[1].outputStart).toBe(5);
    expect(chunks[1].outputCount).toBe(5);

    // Third chunk: tokens 8-14 (2 overlap + remaining)
    expect(chunks[2].tokens).toEqual([9, 10, 11, 12, 13, 14, 15]);
    expect(chunks[2].outputStart).toBe(10);
    expect(chunks[2].outputCount).toBe(5);
  });

  it('should detect chunked format correctly', async () => {
    const { isChunkedFormat, CHUNKED_MAGIC_BYTES } = await import(
      '../src/format/chunked-header.js'
    );
    const { MAGIC_BYTES } = await import('../src/format/header.js');

    expect(isChunkedFormat(new Uint8Array(CHUNKED_MAGIC_BYTES))).toBe(true);
    expect(isChunkedFormat(new Uint8Array(MAGIC_BYTES))).toBe(false);
    expect(isChunkedFormat(new Uint8Array([0, 0, 0, 0]))).toBe(false);
    expect(isChunkedFormat(new Uint8Array([]))).toBe(false);
  });
});

// Print test summary if run with real model
if (runRealTests) {
  console.log('\n========================================');
  console.log('Running chunked compression tests with real model');
  console.log('========================================\n');
} else {
  console.log('\n⚠️  Real model tests skipped.');
  console.log('To run compression ratio comparison tests:');
  console.log('  RWKV_MODEL_PATH=./assets/rwkv-4-pile-169m-uint8.onnx pnpm test test/chunked-compression.test.ts');
  console.log('');
}
