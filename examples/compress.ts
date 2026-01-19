/**
 * Example: Compress and decompress text using the LLM compressor.
 *
 * Usage:
 *   1. Download the model:
 *      curl -L https://huggingface.co/rocca/rwkv-4-pile-web/resolve/main/169m/rwkv-4-pile-169m-uint8.onnx \
 *        -o ./assets/rwkv-4-pile-169m-uint8.onnx
 *
 *   2. Run the example:
 *      npx tsx examples/compress.ts
 *
 *   Or with custom text:
 *      npx tsx examples/compress.ts "Your text here"
 */

import { LLMCompressor } from '../src/index.js';
import * as path from 'path';
import * as fs from 'fs';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_PATH = path.join(__dirname, '../assets/rwkv-4-pile-169m-uint8.onnx');
const TOKENIZER_PATH = path.join(__dirname, '../assets/20B_tokenizer.json');

async function main() {
  // Check if model exists
  if (!fs.existsSync(MODEL_PATH)) {
    console.error('❌ Model not found at:', MODEL_PATH);
    console.log('\nTo download the model, run:');
    console.log(
      '  curl -L https://huggingface.co/rocca/rwkv-4-pile-web/resolve/main/169m/rwkv-4-pile-169m-uint8.onnx -o ./assets/rwkv-4-pile-169m-uint8.onnx'
    );
    process.exit(1);
  }

  // Get text to compress
  const text =
    process.argv[2] ||
    `The quick brown fox jumps over the lazy dog.
This is a test of LLM-based text compression.
It uses RWKV to predict the next token probabilities,
then applies arithmetic coding for efficient compression.`;

  console.log('📝 Original text:');
  console.log('─'.repeat(50));
  console.log(text);
  console.log('─'.repeat(50));
  console.log();

  // Initialize compressor
  console.log('🔄 Loading model...');
  const compressor = new LLMCompressor({
    model: MODEL_PATH,
    tokenizer: TOKENIZER_PATH,
    onProgress: (progress) => {
      if (progress.stage === 'loading') {
        process.stdout.write(`\r  Loading: ${progress.current}/${progress.total}`);
      } else if (progress.stage === 'compressing') {
        process.stdout.write(
          `\r  Compressing: ${progress.current}/${progress.total} tokens`
        );
      } else if (progress.stage === 'decompressing') {
        process.stdout.write(
          `\r  Decompressing: ${progress.current}/${progress.total} tokens`
        );
      }
    },
  });

  await compressor.init();
  console.log('\n✅ Model loaded');
  console.log();

  // Compress
  console.log('🗜️  Compressing...');
  const startCompress = Date.now();
  const result = await compressor.compress(text);
  const compressTime = Date.now() - startCompress;
  console.log();

  console.log('📊 Compression results:');
  console.log(`  Original size:   ${result.originalSize} bytes`);
  console.log(`  Compressed size: ${result.compressedSize} bytes`);
  console.log(`  Compression ratio: ${result.compressionRatio.toFixed(2)}x`);
  console.log(`  Token count: ${result.tokenCount}`);
  console.log(`  Bits per byte: ${((result.compressedSize * 8) / result.originalSize).toFixed(2)}`);
  console.log(`  Compression time: ${(compressTime / 1000).toFixed(1)}s`);
  console.log();

  // Show base64 encoded result
  const base64 = Buffer.from(result.data).toString('base64');
  console.log('🔤 Base64 encoded:');
  console.log('─'.repeat(50));
  console.log(base64);
  console.log('─'.repeat(50));
  console.log();

  // Decompress
  console.log('📤 Decompressing...');
  const startDecompress = Date.now();
  const decompressed = await compressor.decompress(result.data);
  const decompressTime = Date.now() - startDecompress;
  console.log();

  console.log('✅ Decompressed text:');
  console.log('─'.repeat(50));
  console.log(decompressed);
  console.log('─'.repeat(50));
  console.log();

  // Verify
  if (decompressed === text) {
    console.log('✅ Verification: PASSED (decompressed matches original)');
  } else {
    console.log('❌ Verification: FAILED (decompressed does not match original)');
    console.log('Original length:', text.length);
    console.log('Decompressed length:', decompressed.length);
  }

  console.log(`  Decompression time: ${(decompressTime / 1000).toFixed(1)}s`);

  // Cleanup
  compressor.dispose();
}

main().catch((err) => {
  console.error('Error:', err);
  process.exit(1);
});
