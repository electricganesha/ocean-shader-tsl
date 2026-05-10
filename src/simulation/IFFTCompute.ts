import * as THREE from "three";
import {
  Fn,
  vec2,
  vec4,
  add,
  sub,
  mul,
  div,
  sin,
  cos,
  float,
  uniform,
  storage,
  instanceIndex,
  PI,
} from "three/tsl";

/**
 * Radix-2 Cooley-Tukey IFFT implementation in TSL
 * Performs an Inverse FFT to transform frequency-domain spectra into spatial maps.
 */
export const createIFFTCompute = (size: number, inputStorage: any) => {
  const logSize = Math.log2(size);
  
  // Ping-pong buffers for FFT stages
  const outputStorage = storage(
    new THREE.InstancedBufferAttribute(new Float32Array(size * size * 4), 4),
    "vec4",
    size * size
  );
  (outputStorage.value as any).isStorageBufferAttribute = true;

  const pingPongStorage = storage(
    new THREE.InstancedBufferAttribute(new Float32Array(size * size * 4), 4),
    "vec4",
    size * size
  );
  (pingPongStorage.value as any).isStorageBufferAttribute = true;

  /**
   * Bit-reversal pass
   */
  const bitReversePass = (source: any, target: any) => Fn(() => {
    const id = instanceIndex;
    const x = id.mod(size);
    const y = id.div(size);

    let revX = float(0);
    let tempX = x;
    for (let i = 0; i < logSize; i++) {
        revX = revX.mul(2.0).add(tempX.mod(2.0));
        tempX = tempX.div(2.0).floor();
    }
    
    let revY = float(0);
    let tempY = y;
    for (let i = 0; i < logSize; i++) {
        revY = revY.mul(2.0).add(tempY.mod(2.0));
        tempY = tempY.div(2.0).floor();
    }

    const revId = revY.mul(size).add(revX);
    target.element(id).assign(source.element(revId));
  });

  const uStep = uniform(float(2.0));

  /**
   * Cooley-Tukey Butterfly Pass
   */
  const butterflyPass = (horizontal: boolean, source: any, target: any) => Fn(() => {
    const id = instanceIndex;
    const x = id.mod(size);
    const y = id.div(size);
    
    const i = horizontal ? x : y;

    const step = uStep;
    const halfStep = step.div(2.0);
    
    const k = i.mod(step);
    const isLower = k.lessThan(halfStep);
    
    const pairId = isLower.select(id.add(horizontal ? halfStep : halfStep.mul(size)), id.sub(horizontal ? halfStep : halfStep.mul(size)));
    
    const selfValue = source.element(id);
    const pairValue = source.element(pairId);
    
    const angle = float(2.0).mul(PI).mul(k).div(step);
    const w = vec2(cos(angle), sin(angle));
    
    // Process first complex number (x, y)
    const res1_re = pairValue.x.mul(w.x).sub(pairValue.y.mul(w.y));
    const res1_im = pairValue.x.mul(w.y).add(pairValue.y.mul(w.x));
    const rotatedPair1 = vec2(res1_re, res1_im);

    // Process second complex number (z, w)
    const res2_re = pairValue.z.mul(w.x).sub(pairValue.w.mul(w.y));
    const res2_im = pairValue.z.mul(w.y).add(pairValue.w.mul(w.x));
    const rotatedPair2 = vec2(res2_re, res2_im);
    
    let finalValue = isLower.select(
        vec4(
            selfValue.x.add(rotatedPair1.x), selfValue.y.add(rotatedPair1.y),
            selfValue.z.add(rotatedPair2.x), selfValue.w.add(rotatedPair2.y)
        ),
        vec4(
            selfValue.x.sub(rotatedPair1.x), selfValue.y.sub(rotatedPair1.y),
            selfValue.z.sub(rotatedPair2.x), selfValue.w.sub(rotatedPair2.y)
        )
    );

    // NOTE: Do NOT normalize by 1/N or 1/N². Tessendorf ocean IFFTs are unnormalized.
    // The h0(k) spectrum coefficients already encode physical wave amplitudes in metres.
    target.element(id).assign(finalValue);
  });

  // Prepare all passes
  const passes: any[] = [];
  
  // 1. Bit reversal (Input -> Output)
  passes.push({
    compute: bitReversePass(inputStorage, outputStorage)().compute(size * size)
  });

  // 2. Horizontal Butterfly passes
  let currentSource = outputStorage;
  let currentTarget = pingPongStorage;
  
  for (let s = 0; s < logSize; s++) {
      const stepValue = Math.pow(2, s + 1);
      passes.push({
        step: stepValue,
        compute: butterflyPass(true, currentSource, currentTarget)().compute(size * size)
      });
      // Swap
      [currentSource, currentTarget] = [currentTarget, currentSource];
  }

  // 3. Vertical Butterfly passes
  for (let s = 0; s < logSize; s++) {
      const stepValue = Math.pow(2, s + 1);
      passes.push({
        step: stepValue,
        compute: butterflyPass(false, currentSource, currentTarget)().compute(size * size)
      });
      [currentSource, currentTarget] = [currentTarget, currentSource];
  }

  return {
    outputStorage: currentSource,
    uStep,
    passes
  };
};
