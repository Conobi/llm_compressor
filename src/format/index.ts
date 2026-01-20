export {
  type CompressedHeader,
  MAGIC_BYTES,
  FORMAT_VERSION,
  HEADER_SIZE,
  createHeader,
  serializeHeader,
  deserializeHeader,
  combineHeaderAndPayload,
  splitHeaderAndPayload,
} from './header.js';

export {
  type ChunkedHeader,
  type ChunkConfig,
  CHUNKED_MAGIC_BYTES,
  CHUNKED_FORMAT_VERSION,
  CHUNKED_HEADER_BASE_SIZE,
  DEFAULT_CHUNK_CONFIG,
  createChunkedHeader,
  serializeChunkedHeader,
  deserializeChunkedHeader,
  calculateChunkedHeaderSize,
  splitIntoChunks,
  isChunkedFormat,
} from './chunked-header.js';
