//mostly referenced from https://github.com/GarrettGunnell/Water/blob/main/Assets/Shaders/FFTWater.compute

@group(0) @binding(0)
var<uniform> _N: u32;

@group(0) @binding(1)
var<uniform> _Seed: u32;

@group(0) @binding(2)
var<uniform> _LengthScale0: f32;

@group(0) @binding(3)
var<uniform> _LengthScale1: f32;

@group(0) @binding(4)
var<uniform> _LengthScale2: f32;

@group(0) @binding(5)
var<uniform> _LowCutoff: f32;

@group(0) @binding(6)
var<uniform> _HighCutoff: f32;

@group(0) @binding(7)
var<uniform> _Gravity: f32;

@group(0) @binding(8)
var<uniform> _RepeatTime: f32;

@group(0) @binding(9)
var<uniform> _FrameTime: f32;

@group(0) @binding(10)
var<uniform> _Lambda: vec2<f32>;


struct SpectrumParameters {
    scale: f32,
    angle: f32,
    spreadBlend: f32,
    swell: f32,
    alpha: f32,
    peakOmega: f32,
    gamma: f32,
    shortWavesFade: f32,
}
;

@group(0) @binding(11)
var<storage, read> _Spectrums: array<SpectrumParameters>;

@group(0) @binding(12)
var _SpectrumTextures: texture_storage_2d_array<rgba32float, read_write>;

@group(0) @binding(13)
var _InitialSpectrumTextures: texture_storage_2d_array<rgba32float, read_write>;

@group(0) @binding(14)
var _DisplacementTextures: texture_storage_2d_array<rgba32float, read_write>;
@group(0) @binding(15)
var _SlopeTextures: texture_storage_2d_array<rg32float, read_write>;

@group(0) @binding(16)
var<uniform> _Depth: f32;

//fft
const SIZE: u32 = 256;
const LOG_SIZE: u32 = 8;
@group(0) @binding(17)
var _FourierTarget: texture_storage_2d_array<rgba32float, read_write>;

var<workgroup> fftGroupBuffer: array<array<vec4<f32>, SIZE>, 2>;

//foam
@group(0) @binding(18)
var<uniform> _FoamBias: f32;
@group(0) @binding(19)
var<uniform> _FoamDecayRate: f32;
@group(0) @binding(20)
var<uniform> _FoamAdd: f32;
@group(0) @binding(21)
var<uniform> _FoamThreshold: f32;


const PI = 3.141492;

fn hash(n: u32) -> f32 {
    // Integer hash adapted for WGSL
    var m: u32 = (n << 13u) ^ n;
    m = m * (m * m * 15731u + 0x789221u) + 0x1368B9EDu;
    return f32(m & 0x7fffffffu) / f32(0x7fffffffu);
}

fn ComplexMult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn EulerFormula(x: f32) -> vec2<f32> {
    return vec2(cos(x), sin(x));
}

fn UniformToGaussian(u1: f32, u2: f32) -> vec2<f32> {
    let R = sqrt(-2.0 * log(u1));
    let theta = 2.0 * PI * u2;
    return vec2(R * cos(theta), R * sin(theta));
}

fn Dispersion(kMag: f32) -> f32 {
    return sqrt(_Gravity * kMag * tanh(min(kMag * _Depth, 20.0)));
}

fn DispersionDerivative(kMag: f32) -> f32 {
    let th = tanh(min(kMag * _Depth, 20.0));
    let ch = cosh(kMag * _Depth);
    return _Gravity * (_Depth * kMag / ch / ch + th) / Dispersion(kMag) / 2.0;
}

fn NormalizationFactor(s: f32) -> f32 {
    let s2 = s * s;
    let s3 = s2 * s;
    let s4 = s3 * s;
    if (s < 5.0) {
        return -0.000564 * s4 + 0.00776 * s3 - 0.044 * s2 + 0.192 * s + 0.163;
    }

    return -4.80e-08 * s4 + 1.07e-05 * s3 - 9.53e-04 * s2 + 5.90e-02 * s + 3.93e-01;
}

fn Cosine2s(theta: f32, s: f32) -> f32 {
    return NormalizationFactor(s) * pow(abs(cos(0.5 * theta)), 2.0 * s);
}

fn SpreadPower(omega: f32, peakOmega: f32) -> f32 {
    if (omega > peakOmega) {
        return 9.77 * pow(abs(omega / peakOmega), -2.5);
    }
    return 6.97 * pow(abs(omega / peakOmega), 5.0);
}

fn DirectionSpectrum(theta: f32, omega: f32, spectrum: SpectrumParameters) -> f32 {
    let s = SpreadPower(omega, spectrum.peakOmega) + 16.0 * tanh(min(omega / spectrum.peakOmega, 20.0)) * spectrum.swell * spectrum.swell;
    return mix(2.0 / PI * cos(theta) * cos(theta), Cosine2s(theta - spectrum.angle, s), spectrum.spreadBlend);
}

fn TMACorrection(omega: f32) -> f32 {
    let omegaH = omega * sqrt(_Depth / _Gravity);
    if (omegaH <= 1.0) {
        return 0.5 * omegaH * omegaH;
    }
    if (omegaH < 2.0) {
        return 1.0 - 0.5 * (2.0 - omegaH) * (2.0 - omegaH);
    }
    return 1.0;
}

fn JONSWAP(omega: f32, spectrum: SpectrumParameters) -> f32 {
    //select(falseValue, trueValue, condition)
    let sigma = select(0.09, 0.07, omega <= spectrum.peakOmega);

    let r = exp(-(omega - spectrum.peakOmega) * (omega - spectrum.peakOmega) / 2.0 / sigma / sigma / spectrum.peakOmega / spectrum.peakOmega);

    let oneOverOmega = 1.0 / max(omega, 0.001);

    let peakOmegaOverOmega = spectrum.peakOmega / max(omega, 0.001);

    return spectrum.scale * TMACorrection(omega) * spectrum.alpha * _Gravity * _Gravity
        * oneOverOmega * oneOverOmega * oneOverOmega * oneOverOmega * oneOverOmega
        * exp(-1.5 * peakOmegaOverOmega * peakOmegaOverOmega * peakOmegaOverOmega * peakOmegaOverOmega)
        * pow(abs(spectrum.gamma), r);

}

fn ShortWavesFade(kLength: f32, spectrum: SpectrumParameters) -> f32 {
    return exp(-spectrum.shortWavesFade * spectrum.shortWavesFade * kLength * kLength);
}

@compute @workgroup_size(8, 8, 1)
fn initSpectrum(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {

    // textureStore(_InitialSpectrumTextures, invocation_id.xy, 0, vec4(f32(invocation_id.x)/255,f32(invocation_id.y)/255,0.0,1.0));
    // textureStore(_InitialSpectrumTextures, invocation_id.xy, 0, vec4(f32(_Spectrums[0].shortWavesFade),f32(_Spectrums[0].peakOmega)/255,0.0,1.0));
    var seed: u32 = invocation_id.x + _N * invocation_id.y + _N;
    seed += _Seed;

    let lengthScales = array<f32, 3>(f32(_LengthScale0), f32(_LengthScale1), f32(_LengthScale2));

    for (var i: u32 = 0; i < 3; i = i + 1) {
        let halfN = _N / 2;
        let deltaK = 2.0 * PI / lengthScales[i];
        let K = vec2(f32(invocation_id.x) - f32(halfN), f32(invocation_id.y) - f32(halfN)) * deltaK;
        let kLength = length(K);
        // textureStore(_InitialSpectrumTextures, invocation_id.xy, i, vec4(kLength/255.0));

        seed += i + u32(hash(seed)) * 10;
        let uniformRandSamples = vec4(hash(seed), hash(seed * 2), hash(seed * 3), hash(seed * 4));
        let gauss1 = UniformToGaussian(f32(uniformRandSamples.x), f32(uniformRandSamples.y));
        let gauss2 = UniformToGaussian(f32(uniformRandSamples.z), f32(uniformRandSamples.w));
        if (_LowCutoff <= kLength && kLength <= _HighCutoff) {
            let kAngle = atan2(K.y, K.x);
            let omega = Dispersion(kLength);
            let dOmegadk = DispersionDerivative(kLength);

            var spectrum = JONSWAP(omega, _Spectrums[i * 2]) * DirectionSpectrum(kAngle, omega, _Spectrums[i * 2]) * ShortWavesFade(kLength, _Spectrums[i * 2]);
            // let value = vec4( JONSWAP(omega, _Spectrums[i * 2]));

            //     let sigma = select(0.09, 0.07, omega <= _Spectrums[i * 2].peakOmega);
            // let value = vec4( exp(-(omega - _Spectrums[i * 2].peakOmega) * (omega - _Spectrums[i * 2].peakOmega) / 2.0 / sigma / sigma /_Spectrums[i * 2].peakOmega / _Spectrums[i * 2].peakOmega));

            if (_Spectrums[i * 2 + 1].scale > 0) {
                spectrum += JONSWAP(omega, _Spectrums[i * 2 + 1]) * DirectionSpectrum(kAngle, omega, _Spectrums[i * 2 + 1]) * ShortWavesFade(kLength, _Spectrums[i * 2 + 1]);
            }
            let value = vec4(vec2(gauss2.x, gauss1.y) * sqrt(2 * spectrum * abs(dOmegadk) / kLength * deltaK * deltaK), 0.0, 0.0);

            textureStore(_InitialSpectrumTextures, invocation_id.xy, i, value);

        } else {

            textureStore(_InitialSpectrumTextures, invocation_id.xy, i, vec4(0.0));
        }

    }
}

@compute @workgroup_size(8, 8, 1)
fn packSpectrumConjugate(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    for (var i: u32; i < 3; i++) {
        let value: vec4<f32> = textureLoad(_InitialSpectrumTextures, invocation_id.xy, i);
        let valueCon: vec4<f32> = textureLoad(_InitialSpectrumTextures, vec2((_N - invocation_id.x) % _N,(_N - invocation_id.y) % _N), i);
        let packedValue = vec4(value.xy, valueCon.x, -valueCon.y);

        textureStore(_InitialSpectrumTextures, invocation_id.xy, i, packedValue);
    }
}

@compute @workgroup_size(8, 8, 1)
fn updateSpectrum(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {

    let lengthScales = array<f32, 3>(f32(_LengthScale0), f32(_LengthScale1), f32(_LengthScale2));
    for (var i: u32 = 0; i < 3; i = i + 1) {
        let initialSignal: vec4<f32> = textureLoad(_InitialSpectrumTextures, invocation_id.xy, i);
        let h0: vec2<f32> = initialSignal.xy;
        let h0conj: vec2<f32> = initialSignal.zw;

        let halfN = f32(_N / 2);
        let deltaK = 2.0 * PI / lengthScales[i];
        let K = vec2(f32(invocation_id.x) - halfN, f32(invocation_id.y) - halfN) * deltaK;
        let kMag = length(K);
        var kMagRcp = 1.0 / kMag;
        if (kMag < 0.0001) {
            kMagRcp = 10000.0f;
        }

        let w_0 = 2.0 * PI / _RepeatTime;
        let dispersion: f32 = floor(sqrt(_Gravity * kMag) / w_0) * w_0 * _FrameTime;
        let exponent: vec2<f32> = EulerFormula(dispersion);
        let htilde = ComplexMult(h0, exponent) + ComplexMult(h0conj, vec2(exponent.x, -exponent.y));
        let ih = vec2(-htilde.y, htilde.x);

        let displacementX = ih * K.x * K.x * kMagRcp;
        let displacementY = htilde;
        let displacementZ = ih * K.x * K.y * kMagRcp;

        let displacementX_dx = -htilde * K.x * K.x * kMagRcp;
        let displacementY_dx = ih * K.x;
        let displacementZ_dx = -htilde * K.x * K.y * kMagRcp;

        let displacementY_dz = ih * K.y;
        let displacementZ_dz = -htilde * K.xy * K.y * kMagRcp;

        let htildeDisplacementX = vec2(displacementX.x - displacementZ.y, displacementX.y + displacementZ.x);
        let htildeDisplacementZ = vec2(displacementY.x - displacementZ_dx.y, displacementY.y + displacementZ_dx.x);

        let htildeSlopeX = vec2(displacementY_dx.x - displacementY_dz.y, displacementY_dx.y + displacementY_dz.x);
        let htildeSlopeZ = vec2(displacementY_dx.x - displacementZ_dz.y, displacementY_dx.y + displacementZ_dz.x);

        textureStore(_SpectrumTextures, invocation_id.xy, i * 2, vec4(htildeDisplacementX, htildeDisplacementZ));
        textureStore(_SpectrumTextures, invocation_id.xy, i * 2 + 1, vec4(htildeSlopeX, htildeSlopeZ));
    }
}

// fn ComplexExp(a: vec2<f32>) -> vec2<f32> {
//     //kinda sus, but will keep it like this for now
//     let exa = exp(a.x);
//     return vec2(cos(a.y), sin(a.y) * exa);
// }

// fn ComputeTwiddleFactorAndInputIndices(id: vec2<u32>) -> vec4<f32> {
//     let b = _N >> (id.x + 1);
//     let mult = 2.0 * PI * vec2(0.0, 1.0) / f32(_N);
//     let i = (2 * b * (id.y / b) + id.y % b) % _N;
//     let twiddle = ComplexExp(-mult * ((f32(id.y) / f32(b)) * f32(b)));
//     return vec4(twiddle, f32(i), f32(i + b));
// }

struct IndiTwiddles {
    indices: vec2<u32>,
    twiddle: vec2<f32>,
}

fn ButterflyValues(step: u32, index: u32) -> IndiTwiddles {
    let twoPi = 6.28318530718;
    let b = SIZE >> (step + 1);
    let w = index - (index % b);
    let i = (w + index) % SIZE;
    let phase = -twoPi * f32(w) / f32(SIZE);
    let twiX = cos(phase);
    let twiY = -sin(phase);

    return IndiTwiddles(vec2(i,(i + b)), vec2(twiX, twiY));

}

fn FFT(threadIndex: u32, input: vec4<f32>) -> vec4<f32> {
    fftGroupBuffer[0][threadIndex] = input;
    workgroupBarrier();
    var flag: bool = false;

    for (var step: u32 = 0; step < LOG_SIZE; step = step + 1) {
        let inTwi = ButterflyValues(step, threadIndex);

        let v = fftGroupBuffer[u32(flag)][inTwi.indices.y];
        fftGroupBuffer[u32(!flag)][threadIndex] = fftGroupBuffer[u32(flag)][inTwi.indices.x] + vec4(ComplexMult(inTwi.twiddle, v.xy), ComplexMult(inTwi.twiddle, v.zw));

        flag = !flag;

        workgroupBarrier();
    }

    return fftGroupBuffer[u32(flag)][threadIndex];
}
@compute @workgroup_size(SIZE, 1, 1)
fn horizontalFFT(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    for (var i: u32 = 0u; i < 6u; i++) {
        textureStore(_FourierTarget, invocation_id.xy, i, FFT(invocation_id.x, textureLoad(_FourierTarget, invocation_id.xy, i)));
    }
}
@compute @workgroup_size(SIZE, 1, 1)
fn verticalFFT(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    for (var i: u32 = 0u; i < 6u; i++) {
        textureStore(_FourierTarget, invocation_id.yx, i, FFT(invocation_id.x, textureLoad(_FourierTarget, invocation_id.yx, i)));
    }
}

fn Permute(data: vec4<f32>, id: vec3<f32>) -> vec4<f32> {
    return data * (1.0 - 2.0 * ((id.x + id.y) % 2));
}

@compute @workgroup_size(8, 8, 1)
fn assembleMaps(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    for (var i: u32 = 0u; i < 3u; i++) {
        let spec1 = textureLoad(_SpectrumTextures, invocation_id.xy, i * 2);
        let spec2 = textureLoad(_SpectrumTextures, invocation_id.xy, i * 2 + 1);
        let htildeDisplacement = Permute(spec1, vec3<f32>(invocation_id));
        let htildeSlope = Permute(spec2, vec3<f32>(invocation_id));

        let dxdz = htildeDisplacement.rg;
        let dydxz = htildeDisplacement.ba;
        let dyxdyz = htildeSlope.rg;
        let dxxdzz = htildeSlope.ba;

        let jacobian = (1.0 + _Lambda.x * dxxdzz.x) * (1.0 + _Lambda.y * dxxdzz.y) - _Lambda.x * _Lambda.y * dydxz.y * dydxz.y;

        let displacement = vec3(_Lambda.x * dxdz.x, dydxz.x, _Lambda.y * dxdz.y);

        let slopes = dyxdyz.xy / (1.0 + abs(dxxdzz * _Lambda));

        let covariance = slopes.x * slopes.y;

        var foam = textureLoad(_DisplacementTextures, invocation_id.xy, i).a;

        foam *= exp(-_FoamDecayRate);
        foam = saturate(foam);

        let biasedJacobian = max(0.0, -(jacobian - _FoamBias));

        if (biasedJacobian > _FoamThreshold) {
            foam = _FoamAdd * biasedJacobian;
        }

        textureStore(_DisplacementTextures, invocation_id.xy, i, vec4(displacement, foam));
        textureStore(_SlopeTextures, invocation_id.xy, i, vec4(slopes, 0.0, 0.0));
    }
}
