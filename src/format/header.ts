/**
 * Compressed file format header.
 *
 * The header contains metadata needed to decompress the file:
 * - Magic bytes for file identification
 * - Version for format compatibility
 * - Original text length and token count
 * - Model hash for validation
 */

/**
 * Magic bytes identifying an LLM-compressed file.
 * "LLMC" in ASCII.
 */
export const MAGIC_BYTES = new Uint8Array([0x4c, 0x4c, 0x4d, 0x43]);

/**
 * Current format version.
 */
export const FORMAT_VERSION = 1;

/**
 * Header size in bytes.
 */
export const HEADER_SIZE = 25;

/**
 * Compressed file header structure.
 */
export interface CompressedHeader {
  /** Magic bytes: "LLMC" */
  magic: Uint8Array;

  /** Format version */
  version: number;

  /** Original text length in bytes (UTF-8) */
  originalLength: number;

  /** Number of tokens in the sequence */
  tokenCount: number;

  /** Model identifier hash */
  modelHash: number;

  /** Reserved bytes for future use */
  reserved: Uint8Array;
}

/**
 * Create a header for a compressed file.
 */
export function createHeader(
  originalLength: number,
  tokenCount: number,
  modelHash: number
): CompressedHeader {
  return {
    magic: new Uint8Array(MAGIC_BYTES),
    version: FORMAT_VERSION,
    originalLength,
    tokenCount,
    modelHash,
    reserved: new Uint8Array(8),
  };
}

/**
 * Serialize a header to bytes.
 */
export function serializeHeader(header: CompressedHeader): Uint8Array {
  const buffer = new ArrayBuffer(HEADER_SIZE);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  // Magic (4 bytes)
  bytes.set(header.magic, 0);

  // Version (1 byte)
  view.setUint8(4, header.version);

  // Original length (4 bytes, little-endian)
  view.setUint32(5, header.originalLength, true);

  // Token count (4 bytes, little-endian)
  view.setUint32(9, header.tokenCount, true);

  // Model hash (4 bytes, little-endian)
  view.setUint32(13, header.modelHash, true);

  // Reserved (8 bytes)
  bytes.set(header.reserved, 17);

  return bytes;
}

/**
 * Deserialize a header from bytes.
 */
export function deserializeHeader(data: Uint8Array): CompressedHeader {
  if (data.length < HEADER_SIZE) {
    throw new Error(
      `Invalid header: expected ${HEADER_SIZE} bytes, got ${data.length}`
    );
  }

  const view = new DataView(data.buffer, data.byteOffset, HEADER_SIZE);

  // Validate magic bytes
  const magic = data.slice(0, 4);
  if (
    magic[0] !== MAGIC_BYTES[0] ||
    magic[1] !== MAGIC_BYTES[1] ||
    magic[2] !== MAGIC_BYTES[2] ||
    magic[3] !== MAGIC_BYTES[3]
  ) {
    throw new Error('Invalid file format: magic bytes mismatch');
  }

  const version = view.getUint8(4);
  if (version > FORMAT_VERSION) {
    throw new Error(
      `Unsupported format version: ${version} (max supported: ${FORMAT_VERSION})`
    );
  }

  return {
    magic,
    version,
    originalLength: view.getUint32(5, true),
    tokenCount: view.getUint32(9, true),
    modelHash: view.getUint32(13, true),
    reserved: data.slice(17, 25),
  };
}

/**
 * Combine header and payload into a single buffer.
 */
export function combineHeaderAndPayload(
  header: Uint8Array,
  payload: Uint8Array
): Uint8Array {
  const result = new Uint8Array(header.length + payload.length);
  result.set(header, 0);
  result.set(payload, header.length);
  return result;
}

/**
 * Split data into header and payload.
 */
export function splitHeaderAndPayload(
  data: Uint8Array
): { header: CompressedHeader; payload: Uint8Array } {
  const header = deserializeHeader(data);
  const payload = data.slice(HEADER_SIZE);
  return { header, payload };
}
