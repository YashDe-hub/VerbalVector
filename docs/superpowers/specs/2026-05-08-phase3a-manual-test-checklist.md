# Phase 3a Manual Test Checklist

> **Purpose:** Verify the Bluetooth mic integration end-to-end. Automated tests cover hook logic + device-selection contract; these checks cover the OS/browser/hardware boundary that automation cannot reach.

## Prerequisites

- Backend running: `python api.py` (port 5002 by default)
- Frontend running: `cd frontend && npm run dev` (port 5173 by default)
- Chrome 120+ (Safari has known issues with Bluetooth audio inputs)
- Working microphone of any kind

## Smoke Test (no Bluetooth required)

- [ ] Open `http://localhost:5173` in Chrome.
- [ ] On the Analyze view, the Microphone dropdown is visible above the Record/Upload buttons.
- [ ] Initially shows "System default" plus one or more empty-labelled "Microphone (xxxxxxxx)" entries.
- [ ] The hint "Allow microphone access (start a recording once) to see device names." is visible.
- [ ] Click "Using Ray-Ban Meta as your microphone" — pairing instructions expand. Click again — they collapse.
- [ ] Click "Record Audio". Browser prompts for mic permission. Grant it.
- [ ] After permission, the dropdown refreshes — device labels are populated. The hint is gone.
- [ ] Stop recording. Audio uploads and analysis returns successfully.

## Bluetooth Proxy Test (any Bluetooth headset)

- [ ] Pair any Bluetooth headset with the test machine.
- [ ] Confirm the headset is listed as an audio input device in OS sound settings.
- [ ] Refresh `http://localhost:5173`. (Or wait — `devicechange` should auto-refresh.)
- [ ] In the Microphone dropdown, the Bluetooth headset is selectable by name.
- [ ] Select it. Click Record Audio. Speak — verify it captures from the headset, not the laptop mic.
- [ ] Stop recording. Analysis completes successfully.
- [ ] **Hot-plug test:** disconnect the Bluetooth headset (turn off / move out of range). The dropdown should auto-refresh and remove the entry within a couple seconds.

## Hardware Test (when Ray-Ban Meta glasses are available)

- [ ] Pair Ray-Ban Meta glasses via OS Bluetooth settings.
- [ ] Verify they appear as an audio input device.
- [ ] In Chrome, refresh `http://localhost:5173`.
- [ ] Confirm the glasses appear in the Microphone dropdown by name (note the exact name shown — captures Open Question #1 from the design spec).
- [ ] Record a 30-second sample with the glasses selected (presentation mode — wearer speaking). Note environment noise level.
- [ ] Submit. Verify transcript quality is acceptable.
- [ ] Verify analysis completes — feedback, emotion, vector storage.
- [ ] Repeat in noisy environment.
- [ ] Repeat with another speaker nearby (conversation mode).
- [ ] Measure end-to-end latency for a 60s clip.
- [ ] 15-minute continuous recording — note any Bluetooth drops or battery warnings.

## Browser Compatibility

- [ ] Chrome — should work.
- [ ] Firefox — verify codec selection and `enumerateDevices` behavior with Bluetooth.
- [ ] Safari — known issue: Bluetooth audio inputs may not appear in `enumerateDevices()`. Confirm whether still applicable.

## Rollback Criteria

If observed, revert the merge:

- [ ] System default no longer works (regression in default capture)
- [ ] Recording fails when no device is explicitly selected
- [ ] Device dropdown crashes on a browser without `navigator.mediaDevices`
- [ ] Memory leak from stream not being released after refresh
