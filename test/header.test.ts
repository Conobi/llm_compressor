import { describe, it, expect } from 'vitest';
import {
  createHeader,
  serializeHeader,
  deserializeHeader,
  combineHeaderAndPayload,
  splitHeaderAndPayload,
  HEADER_SIZE,
  MAGIC_BYTES,
  FORMAT_VERSION,
} from '../src/format/header.js';

describe('Header', () => {
  it('should create header with correct values', () => {
    const header = createHeader(1000, 50, 0x12345678);

    expect(header.magic).toEqual(MAGIC_BYTES);
    expect(header.version).toBe(FORMAT_VERSION);
    expect(header.originalLength).toBe(1000);
    expect(header.tokenCount).toBe(50);
    expect(header.modelHash).toBe(0x12345678);
    expect(header.reserved.length).toBe(8);
  });

  it('should serialize to correct size', () => {
    const header = createHeader(1000, 50, 0x12345678);
    const bytes = serializeHeader(header);

    expect(bytes.length).toBe(HEADER_SIZE);
  });

  it('should roundtrip through serialize/deserialize', () => {
    const original = createHeader(12345, 678, 0xdeadbeef);
    const bytes = serializeHeader(original);
    const restored = deserializeHeader(bytes);

    expect(restored.magic).toEqual(original.magic);
    expect(restored.version).toBe(original.version);
    expect(restored.originalLength).toBe(original.originalLength);
    expect(restored.tokenCount).toBe(original.tokenCount);
    expect(restored.modelHash).toBe(original.modelHash);
  });

  it('should handle maximum values', () => {
    const original = createHeader(0xffffffff, 0xffffffff, 0xffffffff);
    const bytes = serializeHeader(original);
    const restored = deserializeHeader(bytes);

    expect(restored.originalLength).toBe(0xffffffff);
    expect(restored.tokenCount).toBe(0xffffffff);
    expect(restored.modelHash).toBe(0xffffffff);
  });

  it('should handle zero values', () => {
    const original = createHeader(0, 0, 0);
    const bytes = serializeHeader(original);
    const restored = deserializeHeader(bytes);

    expect(restored.originalLength).toBe(0);
    expect(restored.tokenCount).toBe(0);
    expect(restored.modelHash).toBe(0);
  });

  it('should throw on invalid magic bytes', () => {
    const bytes = new Uint8Array(HEADER_SIZE);
    bytes[0] = 0x00; // Invalid magic

    expect(() => deserializeHeader(bytes)).toThrow('Invalid file format');
  });

  it('should throw on truncated header', () => {
    const bytes = new Uint8Array(10); // Too short

    expect(() => deserializeHeader(bytes)).toThrow('Invalid header');
  });

  it('should throw on unsupported version', () => {
    const header = createHeader(100, 10, 0);
    const bytes = serializeHeader(header);
    bytes[4] = 255; // Set invalid version

    expect(() => deserializeHeader(bytes)).toThrow('Unsupported format version');
  });
});

describe('Header + Payload', () => {
  it('should combine header and payload', () => {
    const header = serializeHeader(createHeader(100, 10, 0x1234));
    const payload = new Uint8Array([1, 2, 3, 4, 5]);

    const combined = combineHeaderAndPayload(header, payload);

    expect(combined.length).toBe(HEADER_SIZE + payload.length);
    expect(combined.slice(0, HEADER_SIZE)).toEqual(header);
    expect(combined.slice(HEADER_SIZE)).toEqual(payload);
  });

  it('should split header and payload', () => {
    const originalHeader = createHeader(100, 10, 0x1234);
    const headerBytes = serializeHeader(originalHeader);
    const payload = new Uint8Array([1, 2, 3, 4, 5]);
    const combined = combineHeaderAndPayload(headerBytes, payload);

    const { header, payload: extractedPayload } = splitHeaderAndPayload(combined);

    expect(header.originalLength).toBe(100);
    expect(header.tokenCount).toBe(10);
    expect(header.modelHash).toBe(0x1234);
    expect(extractedPayload).toEqual(payload);
  });

  it('should handle empty payload', () => {
    const headerBytes = serializeHeader(createHeader(0, 0, 0));
    const payload = new Uint8Array(0);
    const combined = combineHeaderAndPayload(headerBytes, payload);

    const { header, payload: extractedPayload } = splitHeaderAndPayload(combined);

    expect(header.tokenCount).toBe(0);
    expect(extractedPayload.length).toBe(0);
  });
});
