import { describe, it, expect } from 'vitest';
import { BitOutputStream, BitInputStream } from '../src/core/bit-stream.js';

describe('BitOutputStream', () => {
  it('should write and flush single bits correctly', () => {
    const stream = new BitOutputStream();

    // Write 8 bits: 10110100
    stream.writeBit(1);
    stream.writeBit(0);
    stream.writeBit(1);
    stream.writeBit(1);
    stream.writeBit(0);
    stream.writeBit(1);
    stream.writeBit(0);
    stream.writeBit(0);

    const result = stream.toUint8Array();
    expect(result.length).toBe(1);
    expect(result[0]).toBe(0b10110100);
  });

  it('should handle partial bytes with flush', () => {
    const stream = new BitOutputStream();

    // Write 5 bits: 10110
    stream.writeBit(1);
    stream.writeBit(0);
    stream.writeBit(1);
    stream.writeBit(1);
    stream.writeBit(0);
    stream.flush();

    const result = stream.toUint8Array();
    expect(result.length).toBe(1);
    expect(result[0]).toBe(0b10110000); // Padded with zeros
  });

  it('should write multiple bytes', () => {
    const stream = new BitOutputStream();

    // Write 16 bits
    for (let i = 0; i < 16; i++) {
      stream.writeBit(i % 2);
    }

    const result = stream.toUint8Array();
    expect(result.length).toBe(2);
    expect(result[0]).toBe(0b01010101);
    expect(result[1]).toBe(0b01010101);
  });

  it('should write multiple bits at once', () => {
    const stream = new BitOutputStream();
    stream.writeBits(0b11010, 5);
    stream.writeBits(0b101, 3);

    const result = stream.toUint8Array();
    expect(result.length).toBe(1);
    expect(result[0]).toBe(0b11010101);
  });

  it('should track bit count correctly', () => {
    const stream = new BitOutputStream();
    stream.writeBit(1);
    stream.writeBit(0);
    stream.writeBit(1);

    expect(stream.bitCount).toBe(3);
    expect(stream.byteCount).toBe(0);

    stream.writeBits(0, 5);
    expect(stream.bitCount).toBe(8);
    expect(stream.byteCount).toBe(1);
  });
});

describe('BitInputStream', () => {
  it('should read single bits correctly', () => {
    const data = new Uint8Array([0b10110100]);
    const stream = new BitInputStream(data);

    expect(stream.readBit()).toBe(1);
    expect(stream.readBit()).toBe(0);
    expect(stream.readBit()).toBe(1);
    expect(stream.readBit()).toBe(1);
    expect(stream.readBit()).toBe(0);
    expect(stream.readBit()).toBe(1);
    expect(stream.readBit()).toBe(0);
    expect(stream.readBit()).toBe(0);
  });

  it('should read multiple bytes', () => {
    const data = new Uint8Array([0xff, 0x00]);
    const stream = new BitInputStream(data);

    for (let i = 0; i < 8; i++) {
      expect(stream.readBit()).toBe(1);
    }
    for (let i = 0; i < 8; i++) {
      expect(stream.readBit()).toBe(0);
    }
  });

  it('should pad with zeros at end', () => {
    const data = new Uint8Array([0xff]);
    const stream = new BitInputStream(data);

    // Read past end
    for (let i = 0; i < 8; i++) {
      stream.readBit();
    }

    // Should return 0 for bits past end
    expect(stream.readBit()).toBe(0);
    expect(stream.readBit()).toBe(0);
  });

  it('should read multiple bits at once', () => {
    const data = new Uint8Array([0b11010101]);
    const stream = new BitInputStream(data);

    expect(stream.readBits(5)).toBe(0b11010);
    expect(stream.readBits(3)).toBe(0b101);
  });

  it('should track position correctly', () => {
    const data = new Uint8Array([0xff, 0x00]);
    const stream = new BitInputStream(data);

    expect(stream.position).toBe(0);
    expect(stream.size).toBe(16);

    stream.readBit();
    expect(stream.position).toBe(1);

    stream.readBits(7);
    expect(stream.position).toBe(8);
  });
});

describe('BitStream roundtrip', () => {
  it('should preserve data through write/read cycle', () => {
    const outStream = new BitOutputStream();
    const testData = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0];

    for (const bit of testData) {
      outStream.writeBit(bit);
    }
    outStream.flush();

    const inStream = new BitInputStream(outStream.toUint8Array());
    const result: number[] = [];

    for (let i = 0; i < testData.length; i++) {
      result.push(inStream.readBit());
    }

    expect(result).toEqual(testData);
  });

  it('should handle large sequences', () => {
    const outStream = new BitOutputStream();
    const testData: number[] = [];

    // Write 1000 random bits
    for (let i = 0; i < 1000; i++) {
      const bit = Math.random() > 0.5 ? 1 : 0;
      testData.push(bit);
      outStream.writeBit(bit);
    }
    outStream.flush();

    const inStream = new BitInputStream(outStream.toUint8Array());
    const result: number[] = [];

    for (let i = 0; i < testData.length; i++) {
      result.push(inStream.readBit());
    }

    expect(result).toEqual(testData);
  });
});
