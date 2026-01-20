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
 *
 *   Or with chunked compression (parallel decompression):
 *      npx tsx examples/compress.ts --chunked "Your text here"
 *      npx tsx examples/compress.ts --chunked --chunk-size 64 --overlap 8 "Your text here"
 *      npx tsx examples/compress.ts --chunked --workers 4 "Your text here"
 */

import { Worker } from 'worker_threads';
import { LLMCompressor, ChunkedCompressor } from '../src/index.js';
import { deserializeChunkedHeader } from '../src/format/chunked-header.js';
import { BPETokenizer } from '../src/tokenizer/bpe-tokenizer.js';
import * as path from 'path';
import * as fs from 'fs';
import { fileURLToPath } from 'url';
import * as os from 'os';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_PATH = path.join(__dirname, '../assets/rwkv-4-pile-169m-uint8.onnx');
const TOKENIZER_PATH = path.join(__dirname, '../assets/20B_tokenizer.json');
const WORKER_PATH = path.join(__dirname, 'decompress-worker.ts');

interface Options {
  chunked: boolean;
  chunkSize: number;
  overlap: number;
  workers: number;
  text: string;
}

function parseArgs(): Options {
  const args = process.argv.slice(2);
  const options: Options = {
    chunked: false,
    chunkSize: 128,
    overlap: 16,
    workers: Math.min(4, os.cpus().length),
    text: `The quick brown fox jumps over the lazy dog.
This is a test of LLM-based text compression.
It uses RWKV to predict the next token probabilities,
then applies arithmetic coding for efficient compression.`,
  };

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--chunked') {
      options.chunked = true;
    } else if (args[i] === '--chunk-size' && args[i + 1]) {
      options.chunkSize = parseInt(args[++i], 10);
    } else if (args[i] === '--overlap' && args[i + 1]) {
      options.overlap = parseInt(args[++i], 10);
    } else if (args[i] === '--workers' && args[i + 1]) {
      options.workers = parseInt(args[++i], 10);
    } else if (!args[i].startsWith('--')) {
      options.text = args[i];
    }
  }

  return options;
}

async function runSequential(text: string) {
  console.log('🔄 Loading model (sequential mode)...');
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

  compressor.dispose();
}

interface WorkerHandle {
  worker: Worker;
  id: number;
  ready: boolean;
  busy: boolean;
}

interface ChunkResult {
  chunkIndex: number;
  tokens: number[];
}

/**
 * Decompress chunks using worker threads for true parallelism.
 */
async function decompressWithWorkers(
  data: Uint8Array,
  numWorkers: number
): Promise<number[]> {
  const header = deserializeChunkedHeader(data);

  if (header.chunkCount === 0) {
    return [];
  }

  // Extract chunk payloads
  const chunkPayloads: Uint8Array[] = [];
  for (let i = 0; i < header.chunkCount; i++) {
    const start = header.chunkOffsets[i];
    const end = i < header.chunkCount - 1 ? header.chunkOffsets[i + 1] : data.length;
    chunkPayloads.push(data.slice(start, end));
  }

  // Limit workers to number of chunks
  const actualWorkers = Math.min(numWorkers, header.chunkCount);

  console.log(`  Spawning ${actualWorkers} worker threads...`);

  // Create workers
  const workers: WorkerHandle[] = [];
  const workerReadyPromises: Promise<void>[] = [];

  for (let i = 0; i < actualWorkers; i++) {
    const worker = new Worker(WORKER_PATH, {
      workerData: {
        modelPath: MODEL_PATH,
        tokenizerPath: TOKENIZER_PATH,
        workerId: i,
      },
      execArgv: ['--import', 'tsx'],
    });

    const handle: WorkerHandle = {
      worker,
      id: i,
      ready: false,
      busy: false,
    };
    workers.push(handle);

    // Wait for worker to signal ready
    workerReadyPromises.push(
      new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error(`Worker ${i} timed out during initialization`));
        }, 120000); // 2 minute timeout for model loading

        worker.on('message', (msg) => {
          if (msg.type === 'ready') {
            clearTimeout(timeout);
            handle.ready = true;
            resolve();
          }
        });

        worker.on('error', (err) => {
          clearTimeout(timeout);
          reject(err);
        });
      })
    );
  }

  // Wait for all workers to be ready
  const loadStart = Date.now();
  await Promise.all(workerReadyPromises);
  const loadTime = Date.now() - loadStart;
  console.log(`  Workers ready (${(loadTime / 1000).toFixed(1)}s to load models)`);

  // Dispatch chunks to workers
  const results: ChunkResult[] = [];
  let nextChunk = 0;
  const pendingChunks = new Map<number, { resolve: (r: ChunkResult) => void }>();

  // Set up result handlers
  for (const handle of workers) {
    handle.worker.on('message', (msg) => {
      if (msg.type === 'result') {
        const pending = pendingChunks.get(msg.chunkIndex);
        if (pending) {
          pending.resolve({ chunkIndex: msg.chunkIndex, tokens: msg.tokens });
          pendingChunks.delete(msg.chunkIndex);
        }
        handle.busy = false;
      }
    });
  }

  // Process all chunks
  const processChunk = (chunkIndex: number): Promise<ChunkResult> => {
    return new Promise((resolve) => {
      // Find an available worker
      const available = workers.find((w) => w.ready && !w.busy);
      if (!available) {
        throw new Error('No available worker');
      }

      available.busy = true;
      pendingChunks.set(chunkIndex, { resolve });

      available.worker.postMessage({
        type: 'decompress',
        chunkIndex,
        payload: Array.from(chunkPayloads[chunkIndex]), // Convert to array for transfer
        tokenCount: header.chunkTokenCounts[chunkIndex],
        overlapSize: header.overlapSize,
      });
    });
  };

  // Dispatch chunks with worker pool
  const chunkPromises: Promise<ChunkResult>[] = [];

  const dispatchNext = async () => {
    while (nextChunk < header.chunkCount) {
      const available = workers.find((w) => w.ready && !w.busy);
      if (!available) {
        // Wait a bit and try again
        await new Promise((r) => setTimeout(r, 10));
        continue;
      }

      const chunkIndex = nextChunk++;
      process.stdout.write(`\r  Decompressing: chunk ${chunkIndex + 1}/${header.chunkCount}`);
      chunkPromises.push(processChunk(chunkIndex));
    }
  };

  await dispatchNext();
  const chunkResults = await Promise.all(chunkPromises);

  // Sort results by chunk index
  chunkResults.sort((a, b) => a.chunkIndex - b.chunkIndex);

  // Merge results (skip overlap tokens except for first chunk)
  const allTokens: number[] = [];
  for (let i = 0; i < chunkResults.length; i++) {
    const tokens = chunkResults[i].tokens;
    const skipCount = i === 0 ? 0 : header.overlapSize;
    allTokens.push(...tokens.slice(skipCount));
  }

  // Terminate workers
  for (const handle of workers) {
    handle.worker.terminate();
  }

  return allTokens;
}

async function runChunked(text: string, chunkSize: number, overlap: number, numWorkers: number) {
  console.log('🔄 Loading model (chunked mode)...');
  console.log(`  Chunk size: ${chunkSize} tokens, Overlap: ${overlap} tokens`);
  console.log(`  Workers: ${numWorkers}`);

  const compressor = new ChunkedCompressor({
    model: MODEL_PATH,
    tokenizer: TOKENIZER_PATH,
    chunkConfig: { chunkSize, overlapSize: overlap },
    onProgress: (progress) => {
      if (progress.stage === 'loading') {
        process.stdout.write(`\r  Loading: ${progress.current}/${progress.total}`);
      } else if (progress.stage === 'compressing') {
        process.stdout.write(
          `\r  Compressing: chunk ${progress.current + 1}/${progress.total}`
        );
      }
    },
  });

  await compressor.init();
  console.log('\n✅ Model loaded');
  console.log();

  // Compress
  console.log('🗜️  Compressing (chunked)...');
  const startCompress = Date.now();
  const result = await compressor.compress(text);
  const compressTime = Date.now() - startCompress;
  console.log();

  console.log('📊 Compression results:');
  console.log(`  Original size:   ${result.originalSize} bytes`);
  console.log(`  Compressed size: ${result.compressedSize} bytes`);
  console.log(`  Compression ratio: ${result.compressionRatio.toFixed(2)}x`);
  console.log(`  Token count: ${result.tokenCount}`);
  console.log(`  Chunks: ${result.chunkCount}`);
  console.log(`  Bits per byte: ${((result.compressedSize * 8) / result.originalSize).toFixed(2)}`);
  console.log(`  Compression time: ${(compressTime / 1000).toFixed(1)}s`);
  console.log();

  // Show base64 encoded result
  const base64 = Buffer.from(result.data).toString('base64');
  console.log('🔤 Base64 encoded:');
  console.log('─'.repeat(50));
  console.log(base64.length > 200 ? base64.slice(0, 200) + '...' : base64);
  console.log('─'.repeat(50));
  console.log();

  // Load tokenizer for final decoding
  const tokenizer = new BPETokenizer();
  await tokenizer.load(TOKENIZER_PATH);

  // Decompress with workers (true parallel)
  console.log('📤 Decompressing (parallel with workers)...');
  const startDecompress = Date.now();
  const tokens = await decompressWithWorkers(result.data, numWorkers);
  const decompressed = tokenizer.decode(tokens);
  const decompressTime = Date.now() - startDecompress;
  console.log();

  console.log('✅ Decompressed text:');
  console.log('─'.repeat(50));
  console.log(decompressed.length > 500 ? decompressed.slice(0, 500) + '...' : decompressed);
  console.log('─'.repeat(50));
  console.log();

  // Verify
  if (decompressed === text) {
    console.log('✅ Verification: PASSED (decompressed matches original)');
  } else {
    console.log('❌ Verification: FAILED (decompressed does not match original)');
    console.log('Original length:', text.length);
    console.log('Decompressed length:', decompressed.length);
    // Show first difference
    for (let i = 0; i < Math.min(text.length, decompressed.length); i++) {
      if (text[i] !== decompressed[i]) {
        console.log(`First difference at position ${i}:`);
        console.log(`  Expected: "${text.slice(i, i + 20)}..."`);
        console.log(`  Got:      "${decompressed.slice(i, i + 20)}..."`);
        break;
      }
    }
  }

  console.log(`  Decompression time: ${(decompressTime / 1000).toFixed(1)}s (${numWorkers} workers)`);

  compressor.dispose();
}

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

  const options = parseArgs();

  console.log('📝 Original text:');
  console.log('─'.repeat(50));
  console.log(options.text.length > 500 ? options.text.slice(0, 500) + '...' : options.text);
  console.log('─'.repeat(50));
  console.log(`  (${options.text.length} bytes)`);
  console.log();

  if (options.chunked) {
    await runChunked(options.text, options.chunkSize, options.overlap, options.workers);
  } else {
    await runSequential(options.text);
  }
}

main().catch((err) => {
  console.error('Error:', err);
  process.exit(1);
});
