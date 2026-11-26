<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="logo/vobiz-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="logo/vobiz-light.png">
  <img alt="Vobiz" src="logo/vobiz-light.png" height="56">
</picture>
&nbsp;&nbsp;&nbsp;&nbsp;
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="logo/speechmatics-white.png">
  <source media="(prefers-color-scheme: light)" srcset="logo/speechmatics-black.png">
  <img alt="Speechmatics" src="logo/speechmatics-black.png" height="30">
</picture>

# Vobiz + Speechmatics

**Phone calls on Vobiz numbers, with Speechmatics speech recognition and speech synthesis.**

</div>

[Vobiz](https://vobiz.ai/) is a cloud telephony platform: phone numbers, voice calls, SIP trunking
and VobizXML for call control. Its `<Stream>` element sends live call audio to your server over a
WebSocket as 16-bit PCM at 16 kHz, which [Speechmatics](https://www.speechmatics.com/) accepts
without transcoding. Speechmatics provides Speech to Text (STT) on the call and, for the voice
agent, Text to Speech (TTS). Both examples run as a single Python process, with no voice-agent
framework or media server.

## Examples

| Example | What you'll build | Speechmatics SDK |
|---|---|---|
| [**01 Phone Voice Agent**](01-phone-voice-agent/README.md) | Answer phone calls with an AI agent: Speechmatics agent STT detects each caller turn, an OpenAI-compatible LLM (Gemini or OpenAI) writes the reply, and Speechmatics TTS speaks it back into the call, with barge-in | [`speechmatics-agent-stt`](https://pypi.org/project/speechmatics-agent-stt/), [`speechmatics-tts`](https://pypi.org/project/speechmatics-tts/) |
| [**02 Live Call Transcription**](02-live-call-transcription/README.md) | Bridge a caller to a second phone and transcribe both sides live as separate speakers, in the console and on a live web page | [`speechmatics-rt`](https://pypi.org/project/speechmatics-rt/) |

**Start with [01 Phone Voice Agent](01-phone-voice-agent/README.md).**

## What you need

- A Vobiz account with a voice-enabled number, and your Auth ID and Auth Token from the
  [Vobiz console](https://console.vobiz.ai/).
- A Speechmatics API key from the [Speechmatics portal](https://portal.speechmatics.com/).
- Python 3.12 or later.
- A public HTTPS URL for your machine, for example a `cloudflared` or `ngrok` tunnel.
- 02 also needs a second phone to bridge the call to.

Each example's README has the full setup, configuration and troubleshooting.

## Notes

- Outbound calls (`dial.py`, `proxy.py --call`) are real, billed calls; call only numbers you may
  call.
- 02 points your Vobiz number at the proxy on startup unless run with `--no-setup`.
- Keep API keys in `.env` (ignored by git); never commit them.

## Also published in

These examples are also part of the
[Speechmatics Academy](https://github.com/speechmatics/speechmatics-academy/tree/main/integrations/vobiz).

## Documentation

- [Vobiz `<Stream>` XML](https://www.vobiz.ai/docs/xml/stream)
- [Speechmatics Realtime API](https://docs.speechmatics.com/speech-to-text/realtime/quickstart) ·
  [Agent STT](https://docs.speechmatics.com/speech-to-text/agent-stt) ·
  [Text to Speech](https://docs.speechmatics.com/text-to-speech/quickstart)

## License

[MIT](LICENSE). Copyright Speechmatics and Vobiz.
