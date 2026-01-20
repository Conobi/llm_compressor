/**
 * ONNX Runtime backend abstraction layer.
 *
 * Automatically detects and selects the optimal ONNX Runtime backend:
 * - Node.js: onnxruntime-node (native CPU execution)
 * - Browser with WebGPU: webgpu execution provider
 * - Browser with WebGL: webgl execution provider
 * - Fallback: WASM execution provider
 */

import type * as OrtWeb from 'onnxruntime-web';
import { detectPlatform, supportsWebGPU, supportsWebGL } from '../utils/platform.js';

/**
 * Represents an ONNX Runtime backend configuration.
 */
export interface OnnxBackend {
  /** The onnxruntime module to use */
  ort: typeof OrtWeb;
  /** Execution providers in order of preference */
  executionProviders: string[];
  /** Human-readable name for the selected backend */
  backendName: string;
}

/**
 * Attempt to load onnxruntime-node for native execution.
 * Returns null if not available (e.g., in browser or if not installed).
 */
async function tryLoadNodeRuntime(): Promise<typeof OrtWeb | null> {
  try {
    // Dynamic import to avoid bundling issues
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const ortNode = await (Function('return import("onnxruntime-node")')() as Promise<unknown>);
    return ortNode as typeof OrtWeb;
  } catch {
    return null;
  }
}

/**
 * Load onnxruntime-web as fallback.
 */
async function loadWebRuntime(): Promise<typeof OrtWeb> {
  const ortWeb = await import('onnxruntime-web');
  return ortWeb;
}

/**
 * Detect and return the optimal ONNX Runtime backend for the current environment.
 *
 * Selection priority:
 * 1. Node.js: Try onnxruntime-node for native CPU execution (~10-20x faster)
 * 2. Browser with WebGPU: Use webgpu provider (~20-100 tokens/second)
 * 3. Browser with WebGL: Use webgl provider (~5-15 tokens/second)
 * 4. Fallback: Use wasm provider (~2-4 tokens/second)
 */
export async function getOptimalBackend(): Promise<OnnxBackend> {
  const platform = detectPlatform();

  // Node.js environment: try native runtime first
  if (platform.isNode) {
    const ortNode = await tryLoadNodeRuntime();
    if (ortNode) {
      return {
        ort: ortNode,
        executionProviders: ['cpu'],
        backendName: 'node (cpu)',
      };
    }
    // Fallback to WASM in Node.js if onnxruntime-node not available
    const ortWeb = await loadWebRuntime();
    return {
      ort: ortWeb,
      executionProviders: ['wasm'],
      backendName: 'node (wasm fallback)',
    };
  }

  // Browser environment: check GPU capabilities
  const ortWeb = await loadWebRuntime();

  // Try WebGPU first (best performance)
  if (await supportsWebGPU()) {
    return {
      ort: ortWeb,
      executionProviders: ['webgpu', 'wasm'],
      backendName: 'webgpu',
    };
  }

  // Try WebGL as fallback
  if (supportsWebGL()) {
    return {
      ort: ortWeb,
      executionProviders: ['webgl', 'wasm'],
      backendName: 'webgl',
    };
  }

  // WASM fallback (always available)
  return {
    ort: ortWeb,
    executionProviders: ['wasm'],
    backendName: 'wasm',
  };
}
