import { describe, it, expect, beforeAll } from 'vitest';
import { BPETokenizer } from '../src/tokenizer/bpe-tokenizer.js';
import * as path from 'path';

describe('BPETokenizer with real tokenizer', () => {
  let tokenizer: BPETokenizer;

  beforeAll(async () => {
    tokenizer = new BPETokenizer();
    const tokenizerPath = path.join(
      __dirname,
      '../assets/20B_tokenizer.json'
    );
    await tokenizer.load(tokenizerPath);
  });

  it('should load tokenizer with correct vocab size', () => {
    // The 20B tokenizer has 50254 tokens in the vocab (may vary by version)
    expect(tokenizer.getVocabSize()).toBeGreaterThan(50000);
  });

  it('should encode and decode "Hello, world!"', () => {
    const text = 'Hello, world!';
    const tokens = tokenizer.encode(text);
    const decoded = tokenizer.decode(tokens);

    expect(decoded).toBe(text);
    expect(tokens.length).toBeGreaterThan(0);
    expect(tokens.length).toBeLessThan(text.length); // BPE should compress
  });

  it('should encode and decode longer text', () => {
    const text = `The quick brown fox jumps over the lazy dog.
This is a test of the BPE tokenizer with multiple sentences.
It should handle punctuation, numbers (123), and special chars!`;

    const tokens = tokenizer.encode(text);
    const decoded = tokenizer.decode(tokens);

    expect(decoded).toBe(text);
  });

  it('should handle unicode characters', () => {
    const text = 'Hello, 世界! Привет мир! 🌍🚀';
    const tokens = tokenizer.encode(text);
    const decoded = tokenizer.decode(tokens);

    expect(decoded).toBe(text);
  });

  it('should handle code snippets', () => {
    const text = `function hello() {
  console.log("Hello, World!");
  return 42;
}`;

    const tokens = tokenizer.encode(text);
    const decoded = tokenizer.decode(tokens);

    expect(decoded).toBe(text);
  });

  it('should handle markdown', () => {
    const text = `# Heading

This is a **bold** and *italic* text.

\`\`\`javascript
const x = 1;
\`\`\`

- List item 1
- List item 2
`;

    const tokens = tokenizer.encode(text);
    const decoded = tokenizer.decode(tokens);

    expect(decoded).toBe(text);
  });

  it('should produce consistent token counts', () => {
    const text = 'The quick brown fox';
    const tokens1 = tokenizer.encode(text);
    const tokens2 = tokenizer.encode(text);

    expect(tokens1).toEqual(tokens2);
  });

  it('should handle empty string', () => {
    const tokens = tokenizer.encode('');
    expect(tokens).toEqual([]);
    expect(tokenizer.decode([])).toBe('');
  });

  it('should handle whitespace-only string', () => {
    const text = '   \t\n   ';
    const tokens = tokenizer.encode(text);
    const decoded = tokenizer.decode(tokens);

    expect(decoded).toBe(text);
  });
});
