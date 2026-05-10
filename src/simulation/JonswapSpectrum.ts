import * as THREE from "three";
import {
  Fn,
  vec2,
  vec3,
  vec4,
  add,
  dot,
  sin,
  cos,
  exp,
  pow,
  sqrt,
  fract,
  log,
  float,
  uniform,
  storage,
  instanceIndex,
  PI,
} from "three/tsl";

/**
 * JONSWAP Spectrum Compute Shader
 * Generates initial complex amplitudes h0(k) and h0*(-k)
 */
export const createJonswapCompute = (size: number, L: number) => {
  // Use InstancedBufferAttribute for storage node initialization in WebGL fallback
  const BufferAttr = THREE.InstancedBufferAttribute;

  // Uniforms
  const windSpeed = uniform(12.0);
  const fetch = uniform(100000.0); // 100km
  const gamma = uniform(3.3);
  const depth = uniform(100.0);
  const windDir = uniform(vec2(1.0, 0.0));
  const uTime = uniform(0.0);
  const kMin = uniform(0.0);
  const kMax = uniform(9999.0);

  // Storage for the spectrum (Complex amplitudes: vec2 real, vec2 imag)
  // We use a storage buffer of vec4 to store [re, im, re_conj, im_conj]
  const spectrumAttr = new BufferAttr(new Float32Array(size * size * 4), 4);
  (spectrumAttr as any).isStorageBufferAttribute = true;
  const spectrumStorage = storage(spectrumAttr, "vec4", size * size);
  console.log("Spectrum Storage Node:", spectrumStorage);
  console.log("Spectrum Storage Node Value:", (spectrumStorage as any).value);

  // Pseudo-random number generator (Hash function)
  const hash = Fn(([p]: [any]) => {
    const p3 = fract(vec3(p.xyx).mul(float(0.1031)));
    const p3_add = p3.add(dot(p3, p3.yzx.add(float(33.33))));
    return fract(p3_add.x.add(p3_add.y).mul(p3_add.z));
  });

  // Box-Muller for Gaussian Random Numbers
  const gaussian = Fn(([uv]: [any]) => {
    const r1 = hash(uv);
    const r2 = hash(uv.add(vec2(float(123.4), float(567.8))));
    const r = sqrt(log(r1).mul(float(-2.0)));
    const theta = float(2.0).mul(PI).mul(r2);
    return vec2(r.mul(cos(theta)), r.mul(sin(theta)));
  });

  const computeSpectrum = Fn(() => {
    const x = instanceIndex.mod(size);
    const y = instanceIndex.div(size);
    const uvCoord = vec2(float(x), float(y)).div(float(size)).sub(0.5);

    const k = uvCoord.mul(float(2.0).mul(PI).mul(float(size)).div(float(L)));
    const kMag = k.length();

    // Avoid division by zero at k=0
    const g = float(9.81);

    // JONSWAP Parameters
    const omega = sqrt(g.mul(kMag));
    const omegaP = float(22.0).mul(
      pow(g.mul(g).div(windSpeed.mul(fetch)), 1.0 / 3.0),
    );
    const alpha = float(0.076).mul(
      pow(windSpeed.mul(windSpeed).div(fetch.mul(g)), 0.22),
    );

    // Peak enhancement
    const sigma = omega.greaterThan(omegaP).select(float(0.09), float(0.07));
    const r = exp(
      pow(omega.sub(omegaP), 2.0)
        .div(float(2.0).mul(pow(sigma.mul(omegaP), 2.0)))
        .negate(),
    );
    const gammaTerm = pow(gamma, r);

    // Phillips Spectrum part
    const S = alpha
      .mul(g)
      .mul(g)
      .div(pow(omega, 5.0))
      .mul(exp(pow(omegaP.div(omega), 4.0).mul(-1.25)));

    // Apply JONSWAP gamma
    const jonswapBase = S.mul(gammaTerm);

    // Directional Spreading (Simplified Cosine Squared)
    const cosTheta = dot(k.normalize(), windDir.normalize());
    const spreading = pow(cosTheta.clamp(0, 1), 2.0);

    const jonswapDirected = jonswapBase.mul(spreading);

    // Kitaigorodskii depth attenuation
    const wh = omega.mul(sqrt(depth.div(g)));
    const phi = pow(wh, 2.0).div(add(1.0, pow(wh, 2.0)));
    const finalJonswap = jonswapDirected.mul(phi);

    // Complex Amplitudes h0(k)
    const rnd = gaussian(vec2(float(x), float(y)));
    const h0 = rnd.mul(sqrt(finalJonswap.div(2.0)));

    // Hermitian symmetry: also compute h0(-k) at the mirror grid position
    // so the IFFT output is strictly real-valued (Tessendorf ocean requirement).
    const cx = float(size).sub(float(x)).mod(float(size));
    const cy = float(size).sub(float(y)).mod(float(size));
    const uvCoordNeg = vec2(cx, cy).div(float(size)).sub(0.5);
    const kNeg = uvCoordNeg.mul(float(2.0).mul(PI).mul(float(size)).div(float(L)));

    // Directional spreading for -k (opposite wave direction)
    const cosTheta_neg = dot(kNeg.normalize(), windDir.normalize());
    const spreading_neg = pow(cosTheta_neg.clamp(0, 1), 2.0);
    const finalJonswap_neg = jonswapBase.mul(spreading_neg).mul(phi);

    const rndNeg = gaussian(vec2(cx, cy));
    const h0Neg = rndNeg.mul(sqrt(finalJonswap_neg.div(2.0)));

    // Band-limiting and k=0 check
    const isInsideBand = kMag.greaterThan(kMin).and(kMag.lessThan(kMax));
    // Store h0(k) in .xy, conj(h0(-k)) in .zw — guarantees H(-k,t) = conj(H(k,t))
    const finalResult = kMag
      .greaterThan(0.0001)
      .and(isInsideBand)
      .select(vec4(h0.x, h0.y, h0Neg.x, h0Neg.y.negate()), vec4(0));

    // Store in buffer
    spectrumStorage.element(instanceIndex).assign(finalResult);
  });

  const timeEvolvedAttr = new BufferAttr(new Float32Array(size * size * 4), 4);
  (timeEvolvedAttr as any).isStorageBufferAttribute = true;
  const timeEvolvedStorage = storage(timeEvolvedAttr, "vec4", size * size);

  const horizontalAttr = new BufferAttr(new Float32Array(size * size * 4), 4);
  (horizontalAttr as any).isStorageBufferAttribute = true;
  const horizontalStorage = storage(horizontalAttr, "vec4", size * size);

  const jacobianAttr1 = new BufferAttr(new Float32Array(size * size * 4), 4);
  (jacobianAttr1 as any).isStorageBufferAttribute = true;
  const jacobianStorage1 = storage(jacobianAttr1, "vec4", size * size);

  const jacobianAttr2 = new BufferAttr(new Float32Array(size * size * 4), 4);
  (jacobianAttr2 as any).isStorageBufferAttribute = true;
  const jacobianStorage2 = storage(jacobianAttr2, "vec4", size * size);

  const slopesAttr = new BufferAttr(new Float32Array(size * size * 4), 4);
  (slopesAttr as any).isStorageBufferAttribute = true;
  const slopesStorage = storage(slopesAttr, "vec4", size * size);

  const computeTimeEvolution = Fn(() => {
    const id = instanceIndex;
    // .xy = h0(k), .zw = conj(h0(-k))
    const specEntry = spectrumStorage.element(id);
    const h0_k   = specEntry.xy;
    const h0c_mk = specEntry.zw; // conj(h0(-k))

    const x = instanceIndex.mod(size);
    const y = instanceIndex.div(size);
    const uvCoord = vec2(float(x), float(y)).div(float(size)).sub(0.5);
    const k = uvCoord.mul(float(2.0).mul(PI).mul(float(size)).div(float(L)));
    const kMag = k.length();

    const g = float(9.81);
    const omega = sqrt(g.mul(kMag));

    const phase = omega.mul(uTime);

    const cosPhase = cos(phase);
    const sinPhase = sin(phase);

    // Hermitian time evolution: H(k,t) = h0(k)*e^(iωt) + conj(h0(-k))*e^(-iωt)
    // e^(iωt) = cosPhase + i*sinPhase
    const h_plus_re = h0_k.x.mul(cosPhase).sub(h0_k.y.mul(sinPhase));
    const h_plus_im = h0_k.x.mul(sinPhase).add(h0_k.y.mul(cosPhase));
    // e^(-iωt) = cosPhase - i*sinPhase  (applied to conjugate term)
    const h_minus_re = h0c_mk.x.mul(cosPhase).add(h0c_mk.y.mul(sinPhase));
    const h_minus_im = h0c_mk.x.mul(sinPhase).negate().add(h0c_mk.y.mul(cosPhase));

    const h_re = h_plus_re.add(h_minus_re);
    const h_im = h_plus_im.add(h_minus_im);
    const h_t = vec2(h_re, h_im);

    // i * h(k,t) = -im + i*re
    const ih_t = vec2(h_t.y.negate(), h_t.x);

    // Vertical Slopes: dH/dx = i * kx * h(k, t), dH/dz = i * kz * h(k, t)
    const dhdx_t = ih_t.mul(k.x);
    const dhdz_t = ih_t.mul(k.y);

    // Horizontal Displacement: D(k, t) = i * (k/|k|) * h(k, t)
    const kInv = float(1.0).div(kMag.add(0.0001));
    const kUnit = k.mul(kInv);

    const dx_t = ih_t.mul(kUnit.x);
    const dz_t = ih_t.mul(kUnit.y);

    // Partial Derivatives (Frequency Domain)
    // dDx/dx = ik_x * Dx = i * kx * (i * kx/k * h) = -kx^2/k * h
    const dDxdx = h_t.mul(kUnit.x.mul(k.x).negate());
    const dDxdz = h_t.mul(kUnit.x.mul(k.y).negate());
    const dDzdx = h_t.mul(kUnit.y.mul(k.x).negate());
    const dDzdz = h_t.mul(kUnit.y.mul(k.y).negate());

    // Band-limiting
    const isInsideBand = kMag.greaterThan(kMin).and(kMag.lessThan(kMax));
    const finalH = isInsideBand.select(vec4(h_t.x, h_t.y, 0, 0), vec4(0));
    const finalD = isInsideBand.select(
      vec4(dx_t.x, dx_t.y, dz_t.x, dz_t.y),
      vec4(0),
    );
    const finalS = isInsideBand.select(
      vec4(dhdx_t.x, dhdx_t.y, dhdz_t.x, dhdz_t.y),
      vec4(0),
    );

    const finalJ1 = isInsideBand.select(
      vec4(dDxdx.x, dDxdx.y, dDxdz.x, dDxdz.y),
      vec4(0),
    );
    const finalJ2 = isInsideBand.select(
      vec4(dDzdx.x, dDzdx.y, dDzdz.x, dDzdz.y),
      vec4(0),
    );

    timeEvolvedStorage.element(id).assign(finalH);
    horizontalStorage.element(id).assign(finalD);
    slopesStorage.element(id).assign(finalS);

    jacobianStorage1.element(id).assign(finalJ1);
    jacobianStorage2.element(id).assign(finalJ2);
  });

  return {
    computeSpectrum: computeSpectrum().compute(size * size),
    computeTimeEvolution: computeTimeEvolution().compute(size * size),
    spectrumStorage,
    timeEvolvedStorage,
    horizontalStorage,
    jacobianStorage1,
    jacobianStorage2,
    slopesStorage,
    uniforms: {
      windSpeed,
      fetch,
      gamma,
      depth,
      windDir,
      uTime,
      kMin,
      kMax,
    },
  };
};
