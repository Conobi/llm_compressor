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
