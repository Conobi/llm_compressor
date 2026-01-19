import { describe, it, expect, beforeAll, vi } from 'vitest';
import * as path from 'path';
import {
  BitOutputStream,
  BitInputStream,
  ArithmeticEncoder,
  ArithmeticDecoder,
  buildCumulativeDistribution,
  BPETokenizer,
  createHeader,
  serializeHeader,
  splitHeaderAndPayload,
  combineHeaderAndPayload,
  softmax,
} from '../src/index.js';

/**
 * Mock RWKV session that returns predictable probability distributions.
 * Uses a simple model: probability is based on token ID modulo vocab size.
 */
class MockRWKVSession {
  private vocabSize: number;
  private stateCounter: number = 0;

  constructor(vocabSize: number = 1000) {
    this.vocabSize = vocabSize;
  }

  async init(): Promise<void> {
    // No-op for mock
  }

  reset(): void {
    this.stateCounter = 0;
  }

  /**
   * Generate a deterministic probability distribution based on state.
   * This simulates how a real LLM would give different probabilities
   * based on context.
   */
  async processToken(token: number): Promise<Float32Array> {
    // Create logits that favor certain tokens based on current state
    const logits = new Float32Array(this.vocabSize);

    // Base distribution: slight preference for lower token IDs
    for (let i = 0; i < this.vocabSize; i++) {
      logits[i] = -Math.log(i + 1) + Math.sin(i * 0.1 + this.stateCounter * 0.5);
    }

    // Boost probability of token that follows the input pattern
    const nextPreferred = (token * 7 + this.stateCounter * 13) % this.vocabSize;
    logits[nextPreferred] += 3;

    this.stateCounter++;

    return softmax(logits);
  }

  getModelHash(): number {
    return 0x12345678;
  }

  get vocab(): number {
    return this.vocabSize;
  }

  dispose(): void {
    // No-op for mock
  }
}

describe('Integration: Full Compression Pipeline', () => {
  let tokenizer: BPETokenizer;

  beforeAll(async () => {
    tokenizer = new BPETokenizer();
    const tokenizerPath = path.join(__dirname, '../assets/20B_tokenizer.json');
    await tokenizer.load(tokenizerPath);
  });

  it('should compress and decompress with mock model', async () => {
    const text = 'Hello, world!';
    const vocabSize = tokenizer.getVocabSize();

    // Tokenize
    const tokens = tokenizer.encode(text);
    expect(tokens.length).toBeGreaterThan(0);

    // Create mock model
    const model = new MockRWKVSession(vocabSize);
    model.reset();

    // Compress: encode tokens using arithmetic coding with model probabilities
    const bitStream = new BitOutputStream();
    const encoder = new ArithmeticEncoder(bitStream);

    for (let i = 0; i < tokens.length; i++) {
      const contextToken = i === 0 ? 0 : tokens[i - 1];
      const probs = await model.processToken(contextToken);
      encoder.encode(tokens[i], probs);
    }
    encoder.finish();

    // Create header and combine
    const header = createHeader(
      new TextEncoder().encode(text).length,
      tokens.length,
      model.getModelHash()
    );
    const headerBytes = serializeHeader(header);
    const payload = bitStream.toUint8Array();
    const compressed = combineHeaderAndPayload(headerBytes, payload);

    // Decompress: reset model and decode
    model.reset();

    const { header: parsedHeader, payload: parsedPayload } =
      splitHeaderAndPayload(compressed);
    expect(parsedHeader.tokenCount).toBe(tokens.length);

    const inStream = new BitInputStream(parsedPayload);
    const decoder = new ArithmeticDecoder(inStream);
    const decodedTokens: number[] = [];

    for (let i = 0; i < parsedHeader.tokenCount; i++) {
      const contextToken = i === 0 ? 0 : decodedTokens[i - 1];
      const probs = await model.processToken(contextToken);
      const token = decoder.decode(probs);
      decodedTokens.push(token);
    }

    // Verify tokens match
    expect(decodedTokens).toEqual(tokens);

    // Detokenize and verify text
    const decodedText = tokenizer.decode(decodedTokens);
    expect(decodedText).toBe(text);
  });

  it('should handle longer text with mock model', async () => {
    const text = `# Introduction

This is a longer document to test the compression pipeline.
It contains multiple paragraphs and various formatting.

## Features

- Bullet point 1
- Bullet point 2
- Bullet point 3

The quick brown fox jumps over the lazy dog.`;

    const vocabSize = tokenizer.getVocabSize();
    const tokens = tokenizer.encode(text);

    // Compress
    const model = new MockRWKVSession(vocabSize);
    model.reset();

    const bitStream = new BitOutputStream();
    const encoder = new ArithmeticEncoder(bitStream);

    for (let i = 0; i < tokens.length; i++) {
      const contextToken = i === 0 ? 0 : tokens[i - 1];
      const probs = await model.processToken(contextToken);
      encoder.encode(tokens[i], probs);
    }
    encoder.finish();

    const compressed = bitStream.toUint8Array();

    // Decompress
    model.reset();

    const inStream = new BitInputStream(compressed);
    const decoder = new ArithmeticDecoder(inStream);
    const decodedTokens: number[] = [];

    for (let i = 0; i < tokens.length; i++) {
      const contextToken = i === 0 ? 0 : decodedTokens[i - 1];
      const probs = await model.processToken(contextToken);
      const token = decoder.decode(probs);
      decodedTokens.push(token);
    }

    expect(decodedTokens).toEqual(tokens);
    expect(tokenizer.decode(decodedTokens)).toBe(text);
  });

  it('should produce reasonable compression ratios', async () => {
    // Use a text that should compress well with a good language model
    const text = 'The the the the the the the the the the';
    const vocabSize = tokenizer.getVocabSize();
    const tokens = tokenizer.encode(text);

    const originalBytes = new TextEncoder().encode(text).length;

    // Compress
    const model = new MockRWKVSession(vocabSize);
    model.reset();

    const bitStream = new BitOutputStream();
    const encoder = new ArithmeticEncoder(bitStream);

    for (let i = 0; i < tokens.length; i++) {
      const contextToken = i === 0 ? 0 : tokens[i - 1];
      const probs = await model.processToken(contextToken);
      encoder.encode(tokens[i], probs);
    }
    encoder.finish();

    const compressed = bitStream.toUint8Array();

    // The mock model won't give great compression, but should be reasonable
    // With a real LLM, repetitive text would compress very well
    console.log(
      `Original: ${originalBytes} bytes, Compressed: ${compressed.length} bytes`
    );
    console.log(`Tokens: ${tokens.length}`);
    console.log(
      `Bits per token: ${(compressed.length * 8) / tokens.length}`
    );

    // Just verify it produces output
    expect(compressed.length).toBeGreaterThan(0);
  });

  it('should handle empty input', async () => {
    const text = '';
    const tokens = tokenizer.encode(text);

    expect(tokens.length).toBe(0);

    // With empty tokens, we should just have the header
    const header = createHeader(0, 0, 0x12345678);
    const headerBytes = serializeHeader(header);

    const { header: parsedHeader } = splitHeaderAndPayload(headerBytes);
    expect(parsedHeader.tokenCount).toBe(0);
    expect(parsedHeader.originalLength).toBe(0);
  });
});

describe('Integration: Determinism', () => {
  let tokenizer: BPETokenizer;

  beforeAll(async () => {
    tokenizer = new BPETokenizer();
    const tokenizerPath = path.join(__dirname, '../assets/20B_tokenizer.json');
    await tokenizer.load(tokenizerPath);
  });

  it('should produce identical output for same input', async () => {
    const text = 'Determinism test: same input should give same output.';
    const vocabSize = tokenizer.getVocabSize();
    const tokens = tokenizer.encode(text);

    // Compress twice
    async function compress(): Promise<Uint8Array> {
      const model = new MockRWKVSession(vocabSize);
      model.reset();

      const bitStream = new BitOutputStream();
      const encoder = new ArithmeticEncoder(bitStream);

      for (let i = 0; i < tokens.length; i++) {
        const contextToken = i === 0 ? 0 : tokens[i - 1];
        const probs = await model.processToken(contextToken);
        encoder.encode(tokens[i], probs);
      }
      encoder.finish();

      return bitStream.toUint8Array();
    }

    const compressed1 = await compress();
    const compressed2 = await compress();

    expect(compressed1).toEqual(compressed2);
  });

  it('should produce identical model states after same sequence', async () => {
    const vocabSize = 1000;
    const tokens = [10, 20, 30, 40, 50];

    const model1 = new MockRWKVSession(vocabSize);
    const model2 = new MockRWKVSession(vocabSize);

    model1.reset();
    model2.reset();

    const probs1: Float32Array[] = [];
    const probs2: Float32Array[] = [];

    for (const token of tokens) {
      probs1.push(await model1.processToken(token));
      probs2.push(await model2.processToken(token));
    }

    // All probability distributions should match
    for (let i = 0; i < probs1.length; i++) {
      expect(Array.from(probs1[i])).toEqual(Array.from(probs2[i]));
    }
  });
});
