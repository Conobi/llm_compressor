/**
 * Worker thread for parallel chunk decompression.
 *
 * Each worker loads its own model instance and decompresses chunks
 * sent to it via message passing.
 */

import { parentPort, workerData } from 'worker_threads';
// Use built dist files for worker compatibility
import { RWKVSession, BitInputStream, ArithmeticDecoder } from '../dist/index.js';

interface WorkerData {
  modelPath: string;
  tokenizerPath: string;
  workerId: number;
}

interface DecompressRequest {
  type: 'decompress';
  chunkIndex: number;
  payload: Uint8Array;
  tokenCount: number;
  overlapSize: number;
}

interface DecompressResponse {
  type: 'result';
  chunkIndex: number;
  tokens: number[];
  error?: string;
}

interface ReadyMessage {
  type: 'ready';
  workerId: number;
}

const { modelPath, workerId } = workerData as WorkerData;

let model: RWKVSession | null = null;

async function initialize() {
  model = new RWKVSession({ model: modelPath });
  await model.init();

  // Signal ready
  const readyMsg: ReadyMessage = { type: 'ready', workerId };
  parentPort!.postMessage(readyMsg);
}

async function decompressChunk(
  payload: Uint8Array,
  tokenCount: number
): Promise<number[]> {
  if (!model) throw new Error('Model not initialized');

  model.reset();

  const bitStream = new BitInputStream(payload);
  const decoder = new ArithmeticDecoder(bitStream);
  const tokens: number[] = [];

  for (let i = 0; i < tokenCount; i++) {
    const contextToken = i === 0 ? 0 : tokens[i - 1];
    const probs = await model.processToken(contextToken);
    const token = decoder.decode(probs);
    tokens.push(token);
  }

  return tokens;
}

// Handle messages from main thread
parentPort!.on('message', async (msg: DecompressRequest) => {
  if (msg.type === 'decompress') {
    try {
      const tokens = await decompressChunk(
        new Uint8Array(msg.payload),
        msg.tokenCount
      );

      const response: DecompressResponse = {
        type: 'result',
        chunkIndex: msg.chunkIndex,
        tokens,
      };
      parentPort!.postMessage(response);
    } catch (err) {
      const response: DecompressResponse = {
        type: 'result',
        chunkIndex: msg.chunkIndex,
        tokens: [],
        error: String(err),
      };
      parentPort!.postMessage(response);
    }
  }
});

// Initialize on startup
initialize().catch((err) => {
  console.error(`Worker ${workerId} failed to initialize:`, err);
  process.exit(1);
});
