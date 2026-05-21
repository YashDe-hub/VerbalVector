/**
 * Owns the getUserMedia → AudioContext → AudioWorklet pipeline that produces
 * 16kHz 16-bit linear PCM chunks. Used by useLiveStream to feed /api/stream.
 *
 * API shape: start() acquires the mic + audio pipeline and throws on failure.
 * setHandler() wires up where the bytes go. This lets useLiveStream check that
 * the mic is available BEFORE opening the WebSocket — avoiding the case where
 * the server commits a session that never receives audio because the user
 * denied permission.
 *
 * AudioContext + worklet registration are module-scoped singletons so that
 * back-to-back sessions don't hit "processor already registered" or pay the
 * AudioContext-construction cost twice. The mic stream and worklet node are
 * per-session (created in start, destroyed in stop).
 *
 * Not unit-tested — Web Audio API + AudioWorklet do not work in jsdom.
 * Verified via the manual test checklist (smoke + browser path).
 */

export type PcmChunkHandler = (chunk: ArrayBuffer) => void;

export interface PcmAudioCaptureOptions {
  deviceId?: string;
  targetSampleRate?: number; // default 16000
  workletUrl?: string;        // default '/pcm-worklet.js'
}

// Module-scoped singletons — initialized lazily on first start().
let sharedAudioContext: AudioContext | null = null;
let workletRegistered = false;

async function getOrCreateAudioContext(workletUrl: string): Promise<AudioContext> {
  if (!sharedAudioContext || sharedAudioContext.state === 'closed') {
    sharedAudioContext = new AudioContext();
    workletRegistered = false;
  }
  if (!workletRegistered) {
    await sharedAudioContext.audioWorklet.addModule(workletUrl);
    workletRegistered = true;
  }
  return sharedAudioContext;
}

export class PcmAudioCapture {
  private _stream: MediaStream | null = null;
  private _ctx: AudioContext | null = null;
  private _source: MediaStreamAudioSourceNode | null = null;
  private _node: AudioWorkletNode | null = null;
  private _handler: PcmChunkHandler | null = null;

  /**
   * Acquire mic + spin up the audio pipeline. Throws if the user denies mic
   * permission, the device is unavailable, or the worklet fails to load.
   * Bytes flow into the worklet immediately and are discarded until
   * setHandler() is called.
   */
  async start(options: PcmAudioCaptureOptions = {}): Promise<void> {
    const targetRate = options.targetSampleRate ?? 16000;
    const workletUrl = options.workletUrl ?? '/pcm-worklet.js';

    const audioConstraints: MediaTrackConstraints | true = options.deviceId
      ? { deviceId: { exact: options.deviceId } }
      : true;
    this._stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });

    this._ctx = await getOrCreateAudioContext(workletUrl);

    this._source = this._ctx.createMediaStreamSource(this._stream);
    this._node = new AudioWorkletNode(this._ctx, 'pcm-downsampler', {
      processorOptions: { targetSampleRate: targetRate },
    });
    this._node.port.onmessage = (event) => {
      if (this._handler && event.data instanceof ArrayBuffer) {
        this._handler(event.data);
      }
    };

    this._source.connect(this._node);
    // Do NOT connect _node to destination — we don't want to hear ourselves play back.
  }

  /**
   * Wire up where captured PCM bytes go. Pass null to silently drop them.
   * Safe to call before start() — the handler is just stored.
   */
  setHandler(handler: PcmChunkHandler | null): void {
    this._handler = handler;
  }

  async stop(): Promise<void> {
    this._handler = null;
    if (this._node) {
      this._node.port.onmessage = null;
      this._node.disconnect();
      this._node = null;
    }
    if (this._source) {
      this._source.disconnect();
      this._source = null;
    }
    if (this._stream) {
      this._stream.getTracks().forEach((t) => t.stop());
      this._stream = null;
    }
    // Do NOT close sharedAudioContext — it's a singleton reused across sessions.
    this._ctx = null;
  }
}
