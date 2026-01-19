import { describe, it, expect } from 'vitest';
import { BitOutputStream, BitInputStream } from '../src/core/bit-stream.js';
import { ArithmeticEncoder } from '../src/core/arithmetic-encoder.js';
import { ArithmeticDecoder } from '../src/core/arithmetic-decoder.js';
import { buildCumulativeDistribution, findSymbol } from '../src/core/cumulative-dist.js';

describe('Cumulative Distribution', () => {
  it('should build cumulative distribution from probabilities', () => {
    const probs = new Float32Array([0.5, 0.25, 0.25]);
    const cumulative = buildCumulativeDistribution(probs);

    expect(cumulative.length).toBe(4); // n + 1
    expect(cumulative[0]).toBe(0);
    expect(cumulative[3]).toBeGreaterThan(0); // Total should be positive
  });

  it('should find correct symbol for given value', () => {
    const probs = new Float32Array([0.5, 0.25, 0.125, 0.125]);
    const cumulative = buildCumulativeDistribution(probs);
    const total = cumulative[cumulative.length - 1];

    // Values in first half should map to symbol 0
    expect(findSymbol(cumulative, 0)).toBe(0);
    expect(findSymbol(cumulative, Math.floor(total * 0.25))).toBe(0);

    // Values in second quarter should map to symbol 1
    expect(findSymbol(cumulative, Math.floor(total * 0.6))).toBe(1);

    // Last symbol
    expect(findSymbol(cumulative, total - 1)).toBe(3);
  });

  it('should handle uniform distribution', () => {
    const n = 100;
    const probs = new Float32Array(n).fill(1 / n);
    const cumulative = buildCumulativeDistribution(probs);

    expect(cumulative.length).toBe(n + 1);

    // Each symbol should have approximately equal range
    const expectedRange = cumulative[n] / n;
    for (let i = 0; i < n; i++) {
      const range = cumulative[i + 1] - cumulative[i];
      expect(Math.abs(range - expectedRange)).toBeLessThan(expectedRange * 0.1);
    }
  });
});

describe('ArithmeticEncoder/Decoder', () => {
  it('should encode and decode single symbol with uniform distribution', () => {
    const probs = new Float32Array([0.25, 0.25, 0.25, 0.25]);
    const symbol = 2;

    // Encode
    const outStream = new BitOutputStream();
    const encoder = new ArithmeticEncoder(outStream);
    encoder.encode(symbol, probs);
    encoder.finish();

    // Decode
    const inStream = new BitInputStream(outStream.toUint8Array());
    const decoder = new ArithmeticDecoder(inStream);
    const decoded = decoder.decode(probs);

    expect(decoded).toBe(symbol);
  });

  it('should encode and decode multiple symbols with uniform distribution', () => {
    const probs = new Float32Array([0.25, 0.25, 0.25, 0.25]);
    const symbols = [0, 1, 2, 3, 0, 1];

    // Encode
    const outStream = new BitOutputStream();
    const encoder = new ArithmeticEncoder(outStream);
    for (const sym of symbols) {
      encoder.encode(sym, probs);
    }
    encoder.finish();

    // Decode
    const inStream = new BitInputStream(outStream.toUint8Array());
    const decoder = new ArithmeticDecoder(inStream);
    const decoded: number[] = [];
    for (let i = 0; i < symbols.length; i++) {
      decoded.push(decoder.decode(probs));
    }

    expect(decoded).toEqual(symbols);
  });

  it('should handle skewed probability distribution', () => {
    const probs = new Float32Array([0.7, 0.2, 0.1]);
    const symbols = [0, 0, 0, 1, 0, 2, 0, 0];

    // Encode
    const outStream = new BitOutputStream();
    const encoder = new ArithmeticEncoder(outStream);
    for (const sym of symbols) {
      encoder.encode(sym, probs);
    }
    encoder.finish();

    // Decode
    const inStream = new BitInputStream(outStream.toUint8Array());
    const decoder = new ArithmeticDecoder(inStream);
    const decoded: number[] = [];
    for (let i = 0; i < symbols.length; i++) {
      decoded.push(decoder.decode(probs));
    }

    expect(decoded).toEqual(symbols);
  });

  it('should handle time-varying probabilities (like LLM output)', () => {
    // Simulate different probability distributions for each position
    const probSequences = [
      new Float32Array([0.5, 0.3, 0.2]),
      new Float32Array([0.1, 0.8, 0.1]),
      new Float32Array([0.33, 0.33, 0.34]),
      new Float32Array([0.9, 0.05, 0.05]),
    ];
    const symbols = [0, 1, 2, 0];

    // Encode with varying probabilities
    const outStream = new BitOutputStream();
    const encoder = new ArithmeticEncoder(outStream);
    for (let i = 0; i < symbols.length; i++) {
      encoder.encode(symbols[i], probSequences[i]);
    }
    encoder.finish();

    // Decode with same probability sequence
    const inStream = new BitInputStream(outStream.toUint8Array());
    const decoder = new ArithmeticDecoder(inStream);
    const decoded: number[] = [];
    for (let i = 0; i < symbols.length; i++) {
      decoded.push(decoder.decode(probSequences[i]));
    }

    expect(decoded).toEqual(symbols);
  });

  it('should handle large vocabulary (like LLM)', () => {
    const vocabSize = 50277; // GPT-NeoX vocab size
    const probs = new Float32Array(vocabSize);

    // Create a realistic probability distribution (Zipf-like)
    let sum = 0;
    for (let i = 0; i < vocabSize; i++) {
      probs[i] = 1 / (i + 1);
      sum += probs[i];
    }
    // Normalize
    for (let i = 0; i < vocabSize; i++) {
      probs[i] /= sum;
    }

    // Test with a few symbols
    const symbols = [0, 100, 1000, 10000, 50000];

    // Encode
    const outStream = new BitOutputStream();
    const encoder = new ArithmeticEncoder(outStream);
    for (const sym of symbols) {
      encoder.encode(sym, probs);
    }
    encoder.finish();

    // Decode
    const inStream = new BitInputStream(outStream.toUint8Array());
    const decoder = new ArithmeticDecoder(inStream);
    const decoded: number[] = [];
    for (let i = 0; i < symbols.length; i++) {
      decoded.push(decoder.decode(probs));
    }

    expect(decoded).toEqual(symbols);
  });

  it('should achieve near-entropy compression for known distribution', () => {
    // Distribution with known entropy
    const probs = new Float32Array([0.5, 0.25, 0.125, 0.125]);
    // Entropy = 0.5*1 + 0.25*2 + 0.125*3 + 0.125*3 = 1.75 bits/symbol
    const entropy = 1.75;

    // Generate symbols according to distribution
    const numSymbols = 1000;
    const symbols: number[] = [];
    for (let i = 0; i < numSymbols; i++) {
      const r = Math.random();
      if (r < 0.5) symbols.push(0);
      else if (r < 0.75) symbols.push(1);
      else if (r < 0.875) symbols.push(2);
      else symbols.push(3);
    }

    // Encode
    const outStream = new BitOutputStream();
    const encoder = new ArithmeticEncoder(outStream);
    for (const sym of symbols) {
      encoder.encode(sym, probs);
    }
    encoder.finish();

    const compressedBits = outStream.toUint8Array().length * 8;
    const bitsPerSymbol = compressedBits / numSymbols;

    // Should be within 10% of entropy
    expect(bitsPerSymbol).toBeGreaterThan(entropy * 0.9);
    expect(bitsPerSymbol).toBeLessThan(entropy * 1.2);
  });

  it('should handle edge case: very small probabilities', () => {
    const vocabSize = 1000;
    const probs = new Float32Array(vocabSize);

    // One dominant probability, rest are tiny
    probs[0] = 0.99;
    const remaining = 0.01 / (vocabSize - 1);
    for (let i = 1; i < vocabSize; i++) {
      probs[i] = remaining;
    }

    const symbols = [0, 0, 0, 500, 0, 999, 0];

    // Encode
    const outStream = new BitOutputStream();
    const encoder = new ArithmeticEncoder(outStream);
    for (const sym of symbols) {
      encoder.encode(sym, probs);
    }
    encoder.finish();

    // Decode
    const inStream = new BitInputStream(outStream.toUint8Array());
    const decoder = new ArithmeticDecoder(inStream);
    const decoded: number[] = [];
    for (let i = 0; i < symbols.length; i++) {
      decoded.push(decoder.decode(probs));
    }

    expect(decoded).toEqual(symbols);
  });

  it('should handle long sequences without precision loss', () => {
    const probs = new Float32Array([0.4, 0.3, 0.2, 0.1]);
    const numSymbols = 10000;

    // Generate random symbols
    const symbols: number[] = [];
    for (let i = 0; i < numSymbols; i++) {
      symbols.push(Math.floor(Math.random() * 4));
    }

    // Encode
    const outStream = new BitOutputStream();
    const encoder = new ArithmeticEncoder(outStream);
    for (const sym of symbols) {
      encoder.encode(sym, probs);
    }
    encoder.finish();

    // Decode
    const inStream = new BitInputStream(outStream.toUint8Array());
    const decoder = new ArithmeticDecoder(inStream);
    const decoded: number[] = [];
    for (let i = 0; i < symbols.length; i++) {
      decoded.push(decoder.decode(probs));
    }

    expect(decoded).toEqual(symbols);
  });
});
