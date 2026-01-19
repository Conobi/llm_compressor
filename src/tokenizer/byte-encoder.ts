/**
 * Byte encoder/decoder for GPT-2/GPT-NeoX style BPE tokenization.
 *
 * The byte encoder maps each byte (0-255) to a unicode character that is
 * printable and distinct. This allows the BPE vocabulary to use readable
 * string tokens while still being able to represent any byte sequence.
 */

/**
 * Build the byte-to-character mapping used by GPT-2/GPT-NeoX.
 * Returns a Map from byte value to unicode character.
 */
export function buildByteEncoder(): Map<number, string> {
  const byteEncoder = new Map<number, string>();

  // Start with printable ASCII characters that don't need escaping
  // These are: '!' to '~' (33-126) and some other printable chars
  const printableRanges: Array<[number, number]> = [
    [0x21, 0x7e], // ! to ~
    [0xa1, 0xac], // ¡ to ¬
    [0xae, 0xff], // ® to ÿ
  ];

  let n = 0;
  const usedChars = new Set<number>();

  // First pass: assign printable characters to themselves
  for (const [start, end] of printableRanges) {
    for (let b = start; b <= end; b++) {
      byteEncoder.set(b, String.fromCharCode(b));
      usedChars.add(b);
      n++;
    }
  }

  // Second pass: assign remaining bytes to characters starting at 256
  let nextChar = 256;
  for (let b = 0; b < 256; b++) {
    if (!byteEncoder.has(b)) {
      byteEncoder.set(b, String.fromCharCode(nextChar));
      nextChar++;
    }
  }

  return byteEncoder;
}

/**
 * Build the character-to-byte mapping (inverse of byteEncoder).
 */
export function buildByteDecoder(): Map<string, number> {
  const byteEncoder = buildByteEncoder();
  const byteDecoder = new Map<string, number>();

  for (const [byte, char] of byteEncoder) {
    byteDecoder.set(char, byte);
  }

  return byteDecoder;
}

// Pre-computed mappings for efficiency
let cachedByteEncoder: Map<number, string> | null = null;
let cachedByteDecoder: Map<string, number> | null = null;

/**
 * Get the byte encoder (cached).
 */
export function getByteEncoder(): Map<number, string> {
  if (!cachedByteEncoder) {
    cachedByteEncoder = buildByteEncoder();
  }
  return cachedByteEncoder;
}

/**
 * Get the byte decoder (cached).
 */
export function getByteDecoder(): Map<string, number> {
  if (!cachedByteDecoder) {
    cachedByteDecoder = buildByteDecoder();
  }
  return cachedByteDecoder;
}

/**
 * Encode a string to byte-encoded representation.
 * Each UTF-8 byte is mapped to a unicode character.
 */
export function encodeBytes(text: string): string {
  const encoder = new TextEncoder();
  const bytes = encoder.encode(text);
  const byteEncoder = getByteEncoder();

  let result = '';
  for (const byte of bytes) {
    result += byteEncoder.get(byte)!;
  }
  return result;
}

/**
 * Decode a byte-encoded string back to regular text.
 */
export function decodeBytes(encoded: string): string {
  const byteDecoder = getByteDecoder();
  const bytes: number[] = [];

  for (const char of encoded) {
    const byte = byteDecoder.get(char);
    if (byte !== undefined) {
      bytes.push(byte);
    }
  }

  const decoder = new TextDecoder();
  return decoder.decode(new Uint8Array(bytes));
}
