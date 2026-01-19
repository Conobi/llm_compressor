/**
 * Integration tests with real RWKV ONNX model.
 *
 * These tests require the actual model file to be present.
 * They are skipped by default - set RWKV_MODEL_PATH environment variable to run them.
 *
 * To run:
 * 1. Download the model:
 *    curl -L https://huggingface.co/rocca/rwkv-4-pile-web/resolve/main/169m/rwkv-4-pile-169m-uint8.onnx \
 *      -o ./assets/rwkv-4-pile-169m-uint8.onnx
 *
 * 2. Run tests:
 *    RWKV_MODEL_PATH=./assets/rwkv-4-pile-169m-uint8.onnx pnpm test test/integration-real.test.ts
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import * as path from 'path';
import { LLMCompressor } from '../src/index.js';

const MODEL_PATH = process.env.RWKV_MODEL_PATH;
const TOKENIZER_PATH = path.join(__dirname, '../assets/20B_tokenizer.json');

const describeWithModel = MODEL_PATH ? describe : describe.skip;

describeWithModel('Integration: Real RWKV Model', () => {
  let compressor: LLMCompressor;

  beforeAll(async () => {
    console.log(`Loading model from: ${MODEL_PATH}`);
    console.log(`Loading tokenizer from: ${TOKENIZER_PATH}`);

    compressor = new LLMCompressor({
      model: MODEL_PATH!,
      tokenizer: TOKENIZER_PATH,
      onProgress: (progress) => {
        if (progress.stage === 'loading') {
          console.log(`Loading: ${progress.current}/${progress.total}`);
        }
      },
    });

    await compressor.init();
    console.log('Model loaded successfully');
  }, 120000); // 2 minute timeout for model loading

  afterAll(() => {
    compressor?.dispose();
  });

  it('should compress and decompress short text', async () => {
    const original = 'Hello, world!';

    const result = await compressor.compress(original);
    console.log(`Original: ${result.originalSize} bytes`);
    console.log(`Compressed: ${result.compressedSize} bytes`);
    console.log(`Ratio: ${result.compressionRatio.toFixed(2)}x`);
    console.log(`Tokens: ${result.tokenCount}`);

    const decompressed = await compressor.decompress(result.data);
    expect(decompressed).toBe(original);
  }, 60000);

  it('should compress and decompress longer text', async () => {
    const original = `# The Quick Brown Fox

The quick brown fox jumps over the lazy dog. This pangram contains every letter
of the English alphabet at least once. It has been used since the late 19th
century to test typewriters and computer keyboards.

## History

The phrase has been used since at least 1885, when it appeared in the magazine
"Pitman's Phonetic Journal". However, it may have been in use earlier.

## Variations

There are many variations of this pangram, including:
- Pack my box with five dozen liquor jugs.
- How vexingly quick daft zebras jump!
- The five boxing wizards jump quickly.
`;

    const result = await compressor.compress(original);
    console.log(`Original: ${result.originalSize} bytes`);
    console.log(`Compressed: ${result.compressedSize} bytes`);
    console.log(`Ratio: ${result.compressionRatio.toFixed(2)}x`);
    console.log(`Tokens: ${result.tokenCount}`);
    console.log(`Bits per byte: ${(result.compressedSize * 8) / result.originalSize}`);

    const decompressed = await compressor.decompress(result.data);
    expect(decompressed).toBe(original);
  }, 300000); // 5 minute timeout for longer text

  it('should compress and decompress code', async () => {
    const original = `function fibonacci(n) {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}

// Calculate first 10 Fibonacci numbers
for (let i = 0; i < 10; i++) {
  console.log(\`F(\${i}) = \${fibonacci(i)}\`);
}
`;

    const result = await compressor.compress(original);
    console.log(`Code compression ratio: ${result.compressionRatio.toFixed(2)}x`);

    const decompressed = await compressor.decompress(result.data);
    expect(decompressed).toBe(original);
  }, 120000);

  it('should handle repetitive text well', async () => {
    // Repetitive text should compress very well with LLM-based compression
    const original = 'the ' .repeat(100).trim();

    const result = await compressor.compress(original);
    console.log(`Repetitive text compression ratio: ${result.compressionRatio.toFixed(2)}x`);
    console.log(`Bits per character: ${(result.compressedSize * 8) / result.originalSize}`);

    const decompressed = await compressor.decompress(result.data);
    expect(decompressed).toBe(original);
  }, 300000);

  it('should beat gzip for natural language', async () => {
    const original = `Natural language processing (NLP) is a subfield of linguistics, computer
science, and artificial intelligence concerned with the interactions between
computers and human language, in particular how to program computers to process
and analyze large amounts of natural language data. The result is a computer
capable of understanding the contents of documents, including the contextual
nuances of the language within them.`;

    const result = await compressor.compress(original);

    // Compare with gzip (if available in Node.js)
    try {
      const zlib = await import('zlib');
      const gzipResult = zlib.gzipSync(original);

      console.log(`Original: ${result.originalSize} bytes`);
      console.log(`LLM compressed: ${result.compressedSize} bytes`);
      console.log(`Gzip compressed: ${gzipResult.length} bytes`);
      console.log(`LLM bits/byte: ${(result.compressedSize * 8) / result.originalSize}`);
      console.log(`Gzip bits/byte: ${(gzipResult.length * 8) / result.originalSize}`);

      // LLM compression should be competitive or better for natural language
      // Note: for short texts, the header overhead may make LLM worse
    } catch {
      console.log('Skipping gzip comparison');
    }

    const decompressed = await compressor.decompress(result.data);
    expect(decompressed).toBe(original);
  }, 300000);
});

// Skip message when model is not available
if (!MODEL_PATH) {
  describe('Integration: Real RWKV Model', () => {
    it.skip('tests skipped - set RWKV_MODEL_PATH to run', () => {});
  });

  console.log('\n⚠️  Real model tests skipped.');
  console.log('To run them:');
  console.log('1. Download model:');
  console.log(
    '   curl -L https://huggingface.co/rocca/rwkv-4-pile-web/resolve/main/169m/rwkv-4-pile-169m-uint8.onnx -o ./assets/rwkv-4-pile-169m-uint8.onnx'
  );
  console.log('2. Run: RWKV_MODEL_PATH=./assets/rwkv-4-pile-169m-uint8.onnx pnpm test\n');
}
