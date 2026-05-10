declare module "*.css";

interface Navigator {
  readonly gpu: any;
}

declare module "three/webgpu" {
  import { BufferAttribute, Texture } from "three";
  export * from "three";
  export class StorageBufferAttribute extends BufferAttribute {}
  export const WebGPURenderer: any;
  export const NodeMaterial: any;
  export const MeshBasicNodeMaterial: any;
  export const MeshStandardNodeMaterial: any;
  /** WebGPU-only texture that can be bound as a storage texture in compute shaders. */
  export class StorageTexture extends Texture {
    constructor(width?: number, height?: number);
    readonly isStorageTexture: true;
  }
}

declare module "three/tsl" {
  export const Fn: any;
  export const vec2: any;
  export const vec3: any;
  export const vec4: any;
  export const mat3: any;
  export const mul: any;
  export const add: any;
  export const sub: any;
  export const div: any;
  export const dot: any;
  export const cross: any;
  export const normalize: any;
  export const sin: any;
  export const cos: any;
  export const tan: any;
  export const exp: any;
  export const pow: any;
  export const sqrt: any;
  export const log: any;
  export const log2: any;
  export const negate: any;
  export const abs: any;
  export const floor: any;
  export const ceil: any;
  export const fract: any;
  export const clamp: any;
  export const mix: any;
  export const step: any;
  export const smoothstep: any;
  export const max: any;
  export const min: any;
  export const time: any;
  export const uv: any;
  export const positionLocal: any;
  export const positionWorld: any;
  export const normalLocal: any;
  export const normalWorld: any;
  export const cameraPosition: any;
  export const modelViewProjection: any;
  export const reflect: any;
  export const refract: any;
  export const texture: any;
  export const cubeTexture: any;
  export const property: any;
  export const uniform: any;
  export const storage: any;
  export const storageTexture: any;
  export const textureLoad: any;
  export const textureStore: any;
  export const textureSize: any;
  export const textureLevel: any;
  export const maxMipLevel: any;
  export const varying: any;
  export const transformedNormalView: any;
  export const positionViewDirection: any;
  export const normalView: any;
  export const positionView: any;
  export const cameraNear: any;
  export const cameraFar: any;
  export const cameraProjectionMatrix: any;
  export const cameraProjectionMatrixInverse: any;
  export const cameraViewMatrix: any;
  export const viewportUV: any;
  export const viewportSharedTexture: any;
  export const viewportDepthTexture: any;
  export const viewportLinearDepth: any;
  export const viewportSize: any;
  export const float: any;
  export const int: any;
  export const uint: any;
  export const bool: any;
  export const ivec2: any;
  export const ivec3: any;
  export const uvec2: any;
  export const uvec3: any;
  export const color: any;
  export const instanceIndex: any;
  export const workgroupId: any;
  export const localId: any;
  export const PI: any;
  export const Discard: any;
  export const Loop: any;
  export const If: any;
  export const Break: any;
  export const Continue: any;
  export const Return: any;
  export const Var: any;
  export const assign: any;
  export const workgroupBarrier: any;
  export const storageBarrier: any;
}
