// PCM downsampler + Int16 LE encoder.
// Runs in the AudioWorkletGlobalScope. Receives Float32 frames at the
// AudioContext's native sample rate, downsamples to 16kHz via simple
// decimation, converts to Int16 LE, and posts ArrayBuffers to the main
// thread. The main thread sends them as binary WebSocket frames to
// /api/stream.

class PcmDownsamplerProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const targetRate = (options && options.processorOptions && options.processorOptions.targetSampleRate) || 16000;
    this._targetRate = targetRate;
    // sampleRate is a global in the worklet scope — the AudioContext's native rate
    this._ratio = sampleRate / targetRate;
    this._buffer = [];
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel0 = input[0];
    if (!channel0) return true;

    // Decimate: pick every Nth sample where N = nativeRate / targetRate.
    // For 48000 -> 16000 this is N=3. Crude but adequate for speech STT;
    // Deepgram handles further processing server-side.
    //
    // Guard ratio < 1: if the device's native rate is already <= 16kHz
    // (rare — some Bluetooth devices) we'd duplicate samples and produce
    // fake data. Pass through 1:1 instead.
    const step = this._ratio < 1 ? 1 : this._ratio;
    for (let i = 0; i < channel0.length; i += step) {
      const idx = Math.floor(i);
      this._buffer.push(channel0[idx]);
    }

    // Flush in ~250ms chunks at 16kHz = 4000 samples = 8000 bytes
    if (this._buffer.length >= 4000) {
      const flushSamples = this._buffer.splice(0, 4000);
      const int16 = new Int16Array(flushSamples.length);
      for (let i = 0; i < flushSamples.length; i++) {
        const s = Math.max(-1, Math.min(1, flushSamples[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      this.port.postMessage(int16.buffer, [int16.buffer]);
    }

    return true; // keep the processor alive
  }
}

registerProcessor('pcm-downsampler', PcmDownsamplerProcessor);
