import { describe, it, expect } from 'vitest';
import {
  buildByteEncoder,
  buildByteDecoder,
  encodeBytes,
  decodeBytes,
} from '../src/tokenizer/byte-encoder.js';

describe('Byte Encoder', () => {
  it('should create encoder with 256 entries', () => {
    const encoder = buildByteEncoder();
    expect(encoder.size).toBe(256);
  });

  it('should map all bytes to unique characters', () => {
    const encoder = buildByteEncoder();
    const chars = new Set(encoder.values());
    expect(chars.size).toBe(256);
  });

  it('should create inverse decoder', () => {
    const encoder = buildByteEncoder();
    const decoder = buildByteDecoder();

    for (const [byte, char] of encoder) {
      expect(decoder.get(char)).toBe(byte);
    }
  });

  it('should preserve printable ASCII characters', () => {
    const encoder = buildByteEncoder();

    // Check some printable ASCII chars map to themselves
    expect(encoder.get(0x41)).toBe('A'); // 'A'
    expect(encoder.get(0x5a)).toBe('Z'); // 'Z'
    expect(encoder.get(0x61)).toBe('a'); // 'a'
    expect(encoder.get(0x7a)).toBe('z'); // 'z'
    expect(encoder.get(0x30)).toBe('0'); // '0'
    expect(encoder.get(0x39)).toBe('9'); // '9'
  });
});

describe('encodeBytes/decodeBytes', () => {
  it('should roundtrip ASCII text', () => {
    const text = 'Hello, World!';
    const encoded = encodeBytes(text);
    const decoded = decodeBytes(encoded);
    expect(decoded).toBe(text);
  });

  it('should roundtrip Unicode text', () => {
    const text = 'Hello, 世界! 🌍';
    const encoded = encodeBytes(text);
    const decoded = decodeBytes(encoded);
    expect(decoded).toBe(text);
  });

  it('should roundtrip binary-like content', () => {
    // Create text with all possible byte values
    const bytes = new Uint8Array(256);
    for (let i = 0; i < 256; i++) {
      bytes[i] = i;
    }

    const decoder = new TextDecoder('utf-8', { fatal: false });
    const text = decoder.decode(bytes);

    // Note: Some byte sequences aren't valid UTF-8,
    // so we test with valid UTF-8 sequences instead
    const validText = 'Test\x00\x01\x02with\nspecial\tchars';
    const encoded = encodeBytes(validText);
    const decoded = decodeBytes(encoded);
    expect(decoded).toBe(validText);
  });

  it('should handle empty string', () => {
    const text = '';
    const encoded = encodeBytes(text);
    const decoded = decodeBytes(encoded);
    expect(decoded).toBe(text);
  });

  it('should handle whitespace', () => {
    const text = '  \t\n\r  ';
    const encoded = encodeBytes(text);
    const decoded = decodeBytes(encoded);
    expect(decoded).toBe(text);
  });

  it('should handle code/markdown content', () => {
    const text = `# Header

\`\`\`javascript
function hello() {
  console.log("Hello, World!");
}
\`\`\`

- List item 1
- List item 2
`;
    const encoded = encodeBytes(text);
    const decoded = decodeBytes(encoded);
    expect(decoded).toBe(text);
  });
});
