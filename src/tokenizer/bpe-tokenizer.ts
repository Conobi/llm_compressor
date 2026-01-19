import { encodeBytes, decodeBytes } from './byte-encoder.js';

/**
 * Interface for the tokenizer.json configuration format.
 */
export interface TokenizerConfig {
  model: {
    vocab: Record<string, number>;
    merges: string[];
  };
  added_tokens?: Array<{
    id: number;
    content: string;
    single_word: boolean;
    lstrip: boolean;
    rstrip: boolean;
    normalized: boolean;
    special: boolean;
  }>;
}

/**
 * BPE (Byte-Pair Encoding) tokenizer compatible with GPT-NeoX/RWKV.
 *
 * This tokenizer:
 * 1. Converts input text to UTF-8 bytes
 * 2. Maps each byte to a unicode character (byte encoding)
 * 3. Applies BPE merges to compress the representation
 * 4. Returns token IDs from the vocabulary
 */
export class BPETokenizer {
  private vocab: Map<string, number> = new Map();
  private vocabReverse: Map<number, string> = new Map();
  private merges: Map<string, number> = new Map(); // "a b" -> merge rank
  private specialTokens: Map<string, number> = new Map();
  private vocabSize: number = 0;

  /**
   * Load tokenizer configuration from a parsed JSON object.
   */
  loadFromConfig(config: TokenizerConfig): void {
    // Load vocabulary
    this.vocab.clear();
    this.vocabReverse.clear();

    for (const [token, id] of Object.entries(config.model.vocab)) {
      this.vocab.set(token, id);
      this.vocabReverse.set(id, token);
    }

    this.vocabSize = this.vocab.size;

    // Load merges with their priority (rank)
    this.merges.clear();
    for (let i = 0; i < config.model.merges.length; i++) {
      this.merges.set(config.model.merges[i], i);
    }

    // Load special/added tokens
    this.specialTokens.clear();
    if (config.added_tokens) {
      for (const token of config.added_tokens) {
        if (token.special) {
          this.specialTokens.set(token.content, token.id);
        }
      }
    }
  }

  /**
   * Load tokenizer from a JSON string.
   */
  loadFromJSON(json: string): void {
    const config = JSON.parse(json) as TokenizerConfig;
    this.loadFromConfig(config);
  }

  /**
   * Load tokenizer from a URL or file path.
   * Works in both browser and Node.js.
   */
  async load(source: string | URL): Promise<void> {
    let json: string;

    if (typeof window !== 'undefined' && typeof fetch !== 'undefined') {
      // Browser environment
      const response = await fetch(source);
      json = await response.text();
    } else {
      // Node.js environment
      const fs = await import('fs/promises');
      json = await fs.readFile(source.toString(), 'utf-8');
    }

    this.loadFromJSON(json);
  }

  /**
   * Get the vocabulary size.
   */
  getVocabSize(): number {
    return this.vocabSize;
  }

  /**
   * Encode text to token IDs.
   */
  encode(text: string): number[] {
    if (!this.vocab.size) {
      throw new Error('Tokenizer not loaded. Call load() first.');
    }

    // Handle empty string
    if (!text) {
      return [];
    }

    // Convert to byte-encoded representation
    const byteEncoded = encodeBytes(text);

    // Split into individual characters (initial tokens)
    const tokens: string[] = [...byteEncoded];

    // Apply BPE merges
    this.applyBPE(tokens);

    // Convert tokens to IDs
    const ids: number[] = [];
    for (const token of tokens) {
      const id = this.vocab.get(token);
      if (id !== undefined) {
        ids.push(id);
      } else {
        // Handle unknown tokens by encoding each character separately
        // This shouldn't happen with a proper BPE vocabulary
        console.warn(`Unknown token: ${token}`);
        for (const char of token) {
          const charId = this.vocab.get(char);
          if (charId !== undefined) {
            ids.push(charId);
          }
        }
      }
    }

    return ids;
  }

  /**
   * Apply BPE merges to a list of tokens in-place.
   */
  private applyBPE(tokens: string[]): void {
    while (tokens.length > 1) {
      // Find the best pair to merge (lowest rank = highest priority)
      let bestPair: string | null = null;
      let bestRank = Infinity;

      for (let i = 0; i < tokens.length - 1; i++) {
        const pair = `${tokens[i]} ${tokens[i + 1]}`;
        const rank = this.merges.get(pair);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestPair = pair;
        }
      }

      // No more merges possible
      if (bestPair === null) {
        break;
      }

      // Apply the merge
      const [left, right] = bestPair.split(' ');
      const merged = left + right;

      // Replace all occurrences of this pair
      const newTokens: string[] = [];
      let i = 0;
      while (i < tokens.length) {
        if (
          i < tokens.length - 1 &&
          tokens[i] === left &&
          tokens[i + 1] === right
        ) {
          newTokens.push(merged);
          i += 2;
        } else {
          newTokens.push(tokens[i]);
          i++;
        }
      }

      tokens.length = 0;
      tokens.push(...newTokens);
    }
  }

  /**
   * Decode token IDs back to text.
   */
  decode(ids: number[]): string {
    if (!this.vocabReverse.size) {
      throw new Error('Tokenizer not loaded. Call load() first.');
    }

    // Convert IDs to tokens
    let byteEncoded = '';
    for (const id of ids) {
      const token = this.vocabReverse.get(id);
      if (token !== undefined) {
        byteEncoded += token;
      }
    }

    // Decode from byte-encoded representation back to UTF-8 text
    return decodeBytes(byteEncoded);
  }

  /**
   * Get the token string for a given ID.
   */
  idToToken(id: number): string | undefined {
    return this.vocabReverse.get(id);
  }

  /**
   * Get the ID for a given token string.
   */
  tokenToId(token: string): number | undefined {
    return this.vocab.get(token);
  }
}
