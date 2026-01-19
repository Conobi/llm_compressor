/**
 * Platform detection utilities for browser vs Node.js compatibility.
 */

export interface PlatformInfo {
  isNode: boolean;
  isBrowser: boolean;
  supportsWasm: boolean;
  supportsSharedArrayBuffer: boolean;
  availableThreads: number;
}

/**
 * Detect the current runtime platform and capabilities.
 */
export function detectPlatform(): PlatformInfo {
  const isNode =
    typeof process !== 'undefined' &&
    process.versions != null &&
    process.versions.node != null;

  const isBrowser =
    typeof window !== 'undefined' && typeof document !== 'undefined';

  const supportsWasm = typeof WebAssembly !== 'undefined';

  const supportsSharedArrayBuffer = typeof SharedArrayBuffer !== 'undefined';

  let availableThreads = 1;
  if (isNode) {
    try {
      // Dynamic import for Node.js os module
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const os = require('os');
      availableThreads = os.cpus().length;
    } catch {
      availableThreads = 1;
    }
  } else if (isBrowser && typeof navigator !== 'undefined') {
    availableThreads = navigator.hardwareConcurrency ?? 1;
  }

  return {
    isNode,
    isBrowser,
    supportsWasm,
    supportsSharedArrayBuffer,
    availableThreads,
  };
}

/**
 * Check if running in a cross-origin isolated context (required for SharedArrayBuffer).
 */
export function isCrossOriginIsolated(): boolean {
  if (typeof window !== 'undefined') {
    return window.crossOriginIsolated === true;
  }
  return true; // Node.js doesn't have this restriction
}
