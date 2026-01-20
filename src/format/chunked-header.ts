/**
 * Chunked compressed file format header.
 *
 * Extends the base format to support parallel decompression by splitting
 * the token stream into independently decompressible chunks.
 *
 * Format:
 * [Magic: 4 bytes "LLMP"] (P for Parallel)
 * [Version: 1 byte]
 * [Original length: 4 bytes]
 * [Total token count: 4 bytes]
 * [Model hash: 4 bytes]
 * [Chunk count: 2 bytes]
 * [Chunk size: 2 bytes] (tokens per chunk, except last)
 * [Overlap size: 2 bytes] (context tokens from previous chunk)
 * [Reserved: 2 bytes]
 * [Chunk offsets: 4 bytes × chunk_count] (byte offset of each chunk payload)
 * [Chunk token counts: 2 bytes × chunk_count] (tokens in each chunk)
 * [Payload: variable]
 */

/**
 * Magic bytes identifying a chunked LLM-compressed file.
 * "LLMP" in ASCII (P for Parallel).
 */
export const CHUNKED_MAGIC_BYTES = new Uint8Array([0x4c, 0x4c, 0x4d, 0x50]);

/**
 * Current chunked format version.
 */
export const CHUNKED_FORMAT_VERSION = 1;

/**
 * Fixed header size (before chunk metadata).
 */
export const CHUNKED_HEADER_BASE_SIZE = 25;

/**
 * Chunked compressed file header structure.
 */
export interface ChunkedHeader {
  /** Magic bytes: "LLMP" */
  magic: Uint8Array;

  /** Format version */
  version: number;

  /** Original text length in bytes (UTF-8) */
  originalLength: number;

  /** Total number of tokens */
  totalTokenCount: number;

  /** Model identifier hash */
  modelHash: number;

  /** Number of chunks */
  chunkCount: number;

  /** Tokens per chunk (except possibly last) */
  chunkSize: number;

  /** Overlap tokens for context */
  overlapSize: number;

  /** Byte offset of each chunk's payload */
  chunkOffsets: number[];

  /** Token count for each chunk */
  chunkTokenCounts: number[];
}

/**
 * Configuration for chunked compression.
 */
export interface ChunkConfig {
  /** Number of tokens per chunk (default: 128) */
  chunkSize: number;

  /** Number of overlap tokens for context (default: 16) */
  overlapSize: number;
}

/**
 * Default chunk configuration.
 */
export const DEFAULT_CHUNK_CONFIG: ChunkConfig = {
  chunkSize: 128,
  overlapSize: 16,
};

/**
 * Calculate total header size for a given number of chunks.
 */
export function calculateChunkedHeaderSize(chunkCount: number): number {
  // Base header + chunk offsets (4 bytes each) + chunk token counts (2 bytes each)
  return CHUNKED_HEADER_BASE_SIZE + chunkCount * 4 + chunkCount * 2;
}

/**
 * Create a chunked header.
 */
export function createChunkedHeader(
  originalLength: number,
  totalTokenCount: number,
  modelHash: number,
  chunkSize: number,
  overlapSize: number,
  chunkOffsets: number[],
  chunkTokenCounts: number[]
): ChunkedHeader {
  return {
    magic: new Uint8Array(CHUNKED_MAGIC_BYTES),
    version: CHUNKED_FORMAT_VERSION,
    originalLength,
    totalTokenCount,
    modelHash,
    chunkCount: chunkOffsets.length,
    chunkSize,
    overlapSize,
    chunkOffsets,
    chunkTokenCounts,
  };
}

/**
 * Serialize a chunked header to bytes.
 */
export function serializeChunkedHeader(header: ChunkedHeader): Uint8Array {
  const headerSize = calculateChunkedHeaderSize(header.chunkCount);
  const buffer = new ArrayBuffer(headerSize);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  let offset = 0;

  // Magic (4 bytes)
  bytes.set(header.magic, offset);
  offset += 4;

  // Version (1 byte)
  view.setUint8(offset, header.version);
  offset += 1;

  // Original length (4 bytes, little-endian)
  view.setUint32(offset, header.originalLength, true);
  offset += 4;

  // Total token count (4 bytes, little-endian)
  view.setUint32(offset, header.totalTokenCount, true);
  offset += 4;

  // Model hash (4 bytes, little-endian)
  view.setUint32(offset, header.modelHash, true);
  offset += 4;

  // Chunk count (2 bytes, little-endian)
  view.setUint16(offset, header.chunkCount, true);
  offset += 2;

  // Chunk size (2 bytes, little-endian)
  view.setUint16(offset, header.chunkSize, true);
  offset += 2;

  // Overlap size (2 bytes, little-endian)
  view.setUint16(offset, header.overlapSize, true);
  offset += 2;

  // Reserved (2 bytes)
  view.setUint16(offset, 0, true);
  offset += 2;

  // Chunk offsets (4 bytes each)
  for (const chunkOffset of header.chunkOffsets) {
    view.setUint32(offset, chunkOffset, true);
    offset += 4;
  }

  // Chunk token counts (2 bytes each)
  for (const tokenCount of header.chunkTokenCounts) {
    view.setUint16(offset, tokenCount, true);
    offset += 2;
  }

  return bytes;
}

/**
 * Deserialize a chunked header from bytes.
 */
export function deserializeChunkedHeader(data: Uint8Array): ChunkedHeader {
  if (data.length < CHUNKED_HEADER_BASE_SIZE) {
    throw new Error(
      `Invalid chunked header: expected at least ${CHUNKED_HEADER_BASE_SIZE} bytes, got ${data.length}`
    );
  }

  const view = new DataView(data.buffer, data.byteOffset, data.length);
  let offset = 0;

  // Validate magic bytes
  const magic = data.slice(0, 4);
  if (
    magic[0] !== CHUNKED_MAGIC_BYTES[0] ||
    magic[1] !== CHUNKED_MAGIC_BYTES[1] ||
    magic[2] !== CHUNKED_MAGIC_BYTES[2] ||
    magic[3] !== CHUNKED_MAGIC_BYTES[3]
  ) {
    throw new Error('Invalid file format: not a chunked compressed file');
  }
  offset += 4;

  const version = view.getUint8(offset);
  offset += 1;

  if (version > CHUNKED_FORMAT_VERSION) {
    throw new Error(
      `Unsupported chunked format version: ${version} (max supported: ${CHUNKED_FORMAT_VERSION})`
    );
  }

  const originalLength = view.getUint32(offset, true);
  offset += 4;

  const totalTokenCount = view.getUint32(offset, true);
  offset += 4;

  const modelHash = view.getUint32(offset, true);
  offset += 4;

  const chunkCount = view.getUint16(offset, true);
  offset += 2;

  const chunkSize = view.getUint16(offset, true);
  offset += 2;

  const overlapSize = view.getUint16(offset, true);
  offset += 2;

  // Skip reserved
  offset += 2;

  // Read chunk offsets
  const chunkOffsets: number[] = [];
  for (let i = 0; i < chunkCount; i++) {
    chunkOffsets.push(view.getUint32(offset, true));
    offset += 4;
  }

  // Read chunk token counts
  const chunkTokenCounts: number[] = [];
  for (let i = 0; i < chunkCount; i++) {
    chunkTokenCounts.push(view.getUint16(offset, true));
    offset += 2;
  }

  return {
    magic,
    version,
    originalLength,
    totalTokenCount,
    modelHash,
    chunkCount,
    chunkSize,
    overlapSize,
    chunkOffsets,
    chunkTokenCounts,
  };
}

/**
 * Split tokens into chunks with optional overlap.
 *
 * Each chunk (except the first) includes `overlapSize` tokens from the
 * previous chunk as context for better compression.
 *
 * @param tokens - Full token array
 * @param chunkSize - Tokens per chunk (not including overlap)
 * @param overlapSize - Context tokens from previous chunk
 * @returns Array of { tokens, outputStart, outputCount } for each chunk
 */
export function splitIntoChunks(
  tokens: number[],
  chunkSize: number,
  overlapSize: number
): Array<{ tokens: number[]; outputStart: number; outputCount: number }> {
  const chunks: Array<{
    tokens: number[];
    outputStart: number;
    outputCount: number;
  }> = [];

  let outputPosition = 0;

  while (outputPosition < tokens.length) {
    const isFirstChunk = outputPosition === 0;
    const contextStart = isFirstChunk
      ? 0
      : Math.max(0, outputPosition - overlapSize);
    const chunkEnd = Math.min(outputPosition + chunkSize, tokens.length);

    const chunkTokens = tokens.slice(contextStart, chunkEnd);
    const outputStart = outputPosition;
    const outputCount = chunkEnd - outputPosition;

    chunks.push({
      tokens: chunkTokens,
      outputStart,
      outputCount,
    });

    outputPosition = chunkEnd;
  }

  return chunks;
}

/**
 * Check if data is in chunked format by examining magic bytes.
 */
export function isChunkedFormat(data: Uint8Array): boolean {
  if (data.length < 4) return false;
  return (
    data[0] === CHUNKED_MAGIC_BYTES[0] &&
    data[1] === CHUNKED_MAGIC_BYTES[1] &&
    data[2] === CHUNKED_MAGIC_BYTES[2] &&
    data[3] === CHUNKED_MAGIC_BYTES[3]
  );
}
