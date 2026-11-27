#!/usr/bin/env python3
"""
Phone Voice Agent - Vobiz + Speechmatics

A phone voice agent built on Speechmatics agent STT, any OpenAI-compatible LLM
(Gemini by default), and streamed Text to Speech: Speechmatics (default),
ElevenLabs, or Cartesia.

    caller --PSTN--> Vobiz <Stream bidirectional> --WebSocket--> this server
                                                                    |
           Speechmatics agent STT (Linden 1, service-side turn detection)
                                                                    | EndOfTurn
                                                     LLM (streamed, OpenAI-compatible)
                                                                    | one sentence at a time
           TTS: Speechmatics / ElevenLabs / Cartesia (16 kHz PCM, streamed) --playAudio--> caller

Run `python main.py`, then point a Vobiz number or an outbound call at the
answer URL it prints (see dial.py).
"""

import asyncio
import base64
import json
import os
import re
import secrets
import sys
import time
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

import aiohttp
from aiohttp import web
from dotenv import load_dotenv

from speechmatics import tts
from speechmatics.agent_stt import AgentSttAsyncClient, AuthenticationError, ServerMessageType, TranscriptionConfig

load_dotenv()

SM_API_KEY = os.getenv("SPEECHMATICS_API_KEY")
# Optional: the agent STT SDK defaults to the EU endpoint and appends /agent itself.
SM_RT_URL = os.getenv("SPEECHMATICS_RT_URL")
SM_VOICE = tts.Voice(os.getenv("SM_VOICE", "sarah"))  # sarah, theo (UK English) - megan, jack (US English)
LANGUAGE = os.getenv("SM_LANGUAGE", "en")

# Text to Speech provider for the agent's voice: speechmatics (default), elevenlabs, or cartesia.
# Every option streams raw 16-bit PCM at 16 kHz, the format the Stream element plays.
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "speechmatics").strip().lower()
TTS_NAMES = {"speechmatics": "Speechmatics", "elevenlabs": "ElevenLabs", "cartesia": "Cartesia", "gemini-live": "Gemini Live"}
# gemini-live: Gemini Live writes AND speaks the reply in one connection (LLM + voice), from the
# text Speechmatics agent STT produces. It returns 24 kHz PCM, which Vobiz plays as is.
GEMINI_LIVE = TTS_PROVIDER == "gemini-live"
GEMINI_LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.8-live")
GEMINI_VOICE = os.getenv("GEMINI_VOICE", "Aoede")

CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
CARTESIA_MODEL = os.getenv("CARTESIA_MODEL", "sonic-3.6")
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "62ae83ad-4f6a-430b-af41-a9bede9286ca")  # Gemma (UK English)
CARTESIA_VERSION = "2026-08-14"

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # George

# Any OpenAI-compatible chat completions endpoint. The default is Gemini.
LLM_URL = (
    os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
    + "/chat/completions"
)
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
# Model "thinking" delays the first word; a phone agent should answer at once.
# Set LLM_REASONING_EFFORT to an empty value for models that reject the field.
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "minimal")
# Reasoning models such as OpenAI's GPT-5 and later accept only their default temperature:
# set LLM_TEMPERATURE=default (or empty) to leave the field out.
LLM_TEMPERATURE = os.getenv("LLM_TEMPERATURE", "0.6").strip()

PUBLIC_URL = (os.getenv("PUBLIC_URL") or "").rstrip("/")
PORT = int(os.getenv("PORT", "8765"))
# A random path segment keeps the answer URL and the audio WebSocket private to
# this run: the Vobiz audio WebSocket itself carries no signature.
PATH_TOKEN = os.getenv("PATH_TOKEN") or secrets.token_urlsafe(16)

# Each call holds a paid agent STT session and LLM and TTS requests, so cap both
# how many calls run at once and how long one can last.
MAX_CALLS = int(os.getenv("MAX_CALLS", "5"))
MAX_CALL_SECONDS = int(os.getenv("MAX_CALL_SECONDS", "600"))

RATE = 16000
FRAME = 640  # 20 ms of 16-bit mono PCM at 16 kHz
CHUNK = FRAME * 3  # 60 ms per playAudio message keeps barge-in responsive
# Audio sent to the caller: 16 kHz from the TTS providers, 24 kHz from Gemini Live.
OUT_RATE = 24000 if GEMINI_LIVE else RATE
OUT_CHUNK = OUT_RATE * 2 * 60 // 1000
# A sound over the agent pauses it. If no words follow within this many seconds, it was a
# cough or an "mhm", and the agent carries on from the sentence it was cut off in.
RESUME_AFTER_SECONDS = 2.0
# Network delay means audio plays a little later than it is sent: count a sentence as heard
# only once it should have finished this long ago.
PLAYBACK_MARGIN_SECONDS = 0.3

HANGUP_TOKEN = "[END_CALL]"
PROMPT_FILE = Path(__file__).resolve().parent.parent / "assets" / "agent.md"
TTS_NAME = TTS_NAMES.get(TTS_PROVIDER, TTS_PROVIDER)
GREETING = (
    "Hi, I'm Vee, the Vobiz voice assistant. I'm listening with Speechmatics speech to text "
    f"and talking with {TTS_NAME} text to speech. What would you like to know?"
)

# Custom dictionary: brand names agent STT should spell correctly. `sounds_like`
# lists how callers actually say them, including mishearings seen in test calls.
CUSTOM_VOCAB = [
    {"content": "Speechmatics", "sounds_like": ["speech matics", "speech mattics"]},
    {"content": "Vobiz", "sounds_like": ["vo biz", "voh biz", "voi biz", "fabius", "fabies"]},
]

SENTENCE_END = re.compile(r"(.+?[.!?])(\s+|$)", re.S)


# Live page: every log line is also sent to open /live pages as a server-sent event.
EVENTS: list[dict] = []          # recent events, replayed when a page opens
LISTENERS: set[asyncio.Queue] = set()
DEMO_NUMBER = os.getenv("DEMO_NUMBER", "")


def publish(kind: str, text: str = "", **extra) -> None:
    """Send one event to every open live page."""
    event = {"t": time.time(), "kind": kind, "text": text, **extra}
    EVENTS.append(event)
    del EVENTS[:-300]
    for queue in list(LISTENERS):
        queue.put_nowait(event)


def log(*parts: object) -> None:
    """Print a timestamped log line and publish it to the live page."""
    print(time.strftime("%H:%M:%S"), *parts, flush=True)
    line = " ".join(str(p) for p in parts)
    kind, _, text = line.partition(" ")
    publish(kind, text.strip())


def answer_xml() -> str:
    """VobizXML that connects the call to this server over a bidirectional stream."""
    stream_url = f"{PUBLIC_URL.replace('https://', 'wss://')}/ws/{PATH_TOKEN}"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<Response>\n"
        f'  <Stream bidirectional="true" keepCallAlive="true" audioTrack="inbound" '
        f'contentType="audio/x-l16;rate={RATE}">{escape(stream_url)}</Stream>\n'
        "</Response>\n"
    )


def system_prompt() -> str:
    """Load the agent's instructions from assets/agent.md."""
    text = PROMPT_FILE.read_text(encoding="utf-8")
    if GEMINI_LIVE:
        text = text.replace("and end your\nreply with {HANGUP_TOKEN}.", "and then call the end_call tool. Say nothing after calling it.")
        text += ("\nIn this setup one Gemini Live model writes and speaks your replies; there is no "
                 "separate text to speech step. The caller's words reach you as text from Speechmatics.\n")
    return text.replace("{HANGUP_TOKEN}", HANGUP_TOKEN).replace("{TTS_NAME}", TTS_NAME)


@dataclass
class Sentence:
    """One spoken sentence, kept so a paused reply can be replayed."""

    text: str
    pcm: bytearray = field(default_factory=bytearray)
    sent: int = 0  # bytes sent to Vobiz since the last clearAudio
    ends_at: float = 0.0  # estimated time its last sent byte finishes playing
    complete: bool = False  # Text to Speech has delivered all of it


class Call:
    """One phone call: Vobiz audio in, agent STT, LLM, TTS, Vobiz audio out."""

    def __init__(self, vobiz: web.WebSocketResponse, app: web.Application):
        self.vobiz = vobiz
        self.http: aiohttp.ClientSession = app["http"]
        self.tts: tts.AsyncClient = app["tts"]
        self.stt: AgentSttAsyncClient | None = None
        self.stream_id: str | None = None
        self.history: list[dict] = [{"role": "system", "content": app["prompt"]}]
        self.turn_text: list[str] = []
        self.reply_task: asyncio.Task | None = None
        self.tasks: set[asyncio.Task] = set()
        self.speaking = False
        self.hangup_after_speech = False
        self.turn_ended_at = 0.0
        # The greeting or reply being spoken, sentence by sentence.
        self.sentences: list[Sentence] = []
        self.speech_done = False  # every sentence of it has been written
        self.checkpoint_name: str | None = None  # the end-of-speech checkpoint sent to Vobiz
        self.checkpoints = 0
        self.play_end = 0.0  # when the audio sent so far should finish playing
        self.paused = False
        self.resume_from = 0  # first sentence the caller has not fully heard
        self.resume_task: asyncio.Task | None = None
        self.send_lock = asyncio.Lock()  # keeps playAudio in order across tasks
        # Gemini Live (TTS_PROVIDER=gemini-live)
        self.prompt: str = app["prompt"]
        self.gl_cm = None
        self.gl = None
        self.gl_generating = False  # a reply is being generated
        self.gl_drop = False  # discard the rest of the current generation (caller cut in)
        self.gl_done: asyncio.Event | None = None
        self.gl_sentence: Sentence | None = None
        self.gl_text = ""
        self.gl_carry = b""
        self.gl_skip_ends = 0  # ends of replaced generations still to come from Gemini

    def spawn(self, coro) -> asyncio.Task:
        """Run a coroutine in the background and keep a reference until it finishes."""
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    # ---------- Vobiz side ----------
    async def send_vobiz(self, msg: dict) -> None:
        """Send one JSON event to Vobiz if the call is still connected."""
        if not self.vobiz.closed:
            await self.vobiz.send_str(json.dumps(msg))

    async def play_pcm(self, pcm: bytes) -> None:
        """Send 16 kHz PCM to the caller as base64 playAudio messages."""
        # playAudio carries only the event and the media. Vobiz plays these messages
        # on the stream they arrive on.
        for i in range(0, len(pcm), OUT_CHUNK):
            await self.send_vobiz(
                {
                    "event": "playAudio",
                    "media": {
                        "contentType": "audio/x-l16",
                        "sampleRate": OUT_RATE,
                        "payload": base64.b64encode(pcm[i : i + OUT_CHUNK]).decode(),
                    },
                }
            )

    def start_speech(self) -> None:
        """Begin a new greeting or reply: forget the previous one's audio."""
        self.sentences = []
        self.speech_done = False
        self.checkpoint_name = None

    async def flush(self) -> None:
        """Send all buffered audio not yet sent, in order, then the end-of-speech checkpoint.

        Nothing is sent while the agent is paused; the audio stays buffered for a replay.
        """
        async with self.send_lock:
            while not self.paused:
                sentence = next((s for s in self.sentences if s.sent < len(s.pcm)), None)
                if sentence is None:
                    break
                chunk = bytes(sentence.pcm[sentence.sent : sentence.sent + OUT_CHUNK])
                sentence.sent += len(chunk)
                # Vobiz plays audio back to back, so track when what was sent will have played.
                self.play_end = max(time.monotonic(), self.play_end) + len(chunk) / (OUT_RATE * 2)
                sentence.ends_at = self.play_end
                self.speaking = True
                await self.play_pcm(chunk)
            if self.speech_done and self.sentences and self.checkpoint_name is None and not self.paused:
                # Vobiz answers with playedStream once everything queued before it has played.
                self.checkpoints += 1
                self.checkpoint_name = f"speech-{self.checkpoints}"
                await self.send_vobiz({"event": "checkpoint", "streamId": self.stream_id, "name": self.checkpoint_name})

    async def pause(self) -> None:
        """The caller made a sound over the agent: stop the audio, but keep the reply.

        SpeechStarted fires on any voice activity, including a cough or an "mhm". Only
        recognized words (StartOfTurn) drop the reply; otherwise it carries on after
        RESUME_AFTER_SECONDS from the sentence it was cut off in.
        """
        if self.resume_task:
            self.resume_task.cancel()  # each new sound restarts the wait
        self.resume_task = self.spawn(self.resume_after(RESUME_AFTER_SECONDS))
        if self.paused:
            return
        self.paused = True
        self.speaking = False
        now = time.monotonic()
        self.resume_from = next(
            (
                i
                for i, s in enumerate(self.sentences)
                if not (s.complete and s.sent == len(s.pcm) and s.ends_at + PLAYBACK_MARGIN_SECONDS <= now)
            ),
            len(self.sentences),
        )
        log("PAUSE     caller made a sound over the agent, pausing until words follow")
        async with self.send_lock:
            await self.send_vobiz({"event": "clearAudio", "streamId": self.stream_id})
        self.play_end = 0.0

    async def resume_after(self, seconds: float) -> None:
        """No words followed the sound: replay from the sentence that was cut off."""
        await asyncio.sleep(seconds)
        if not self.paused:
            return
        for sentence in self.sentences[self.resume_from :]:
            sentence.sent, sentence.ends_at = 0, 0.0
        self.checkpoint_name = None
        self.paused = False
        if self.resume_from < len(self.sentences):
            log(f"RESUME    no words followed, carrying on from: {self.sentences[self.resume_from].text}")
        else:
            log("RESUME    no words followed, and the reply had already been heard")
        await self.flush()

    async def barge_in(self) -> None:
        """The caller said words: drop the reply, and stop its audio if it is playing.

        A reply that has not started speaking yet is replaced too: the caller has
        started a new turn, so the reply is out of date.
        """
        pending = self.reply_task is not None and not self.reply_task.done()
        was_playing = self.speaking
        if was_playing or self.paused:
            log("BARGE-IN  caller spoke over the agent, dropping the reply")
        elif pending:
            log("TURN      caller started a new turn before the reply was spoken, replacing it")
        else:
            return
        if pending:
            self.reply_task.cancel()
        if GEMINI_LIVE and self.gl_generating:
            self.gl_drop = True  # the rest of this Gemini reply is out of date
        if self.resume_task:
            self.resume_task.cancel()
        self.paused = self.speaking = self.hangup_after_speech = False
        self.start_speech()
        if was_playing:  # a paused reply's audio was already cleared
            async with self.send_lock:
                await self.send_vobiz({"event": "clearAudio", "streamId": self.stream_id})
        self.play_end = 0.0

    # ---------- Speechmatics agent STT ----------
    async def start_stt(self) -> None:
        """Open this call's agent STT session."""
        self.stt = AgentSttAsyncClient(
            api_key=SM_API_KEY,
            url=SM_RT_URL,
            app="speechmatics-academy/vobiz-phone-voice-agent",
            # 16 kHz signed 16-bit PCM is the SDK default and what the Stream element asks Vobiz for.
            # `model` defaults to Linden 1; turns are closed by the service's VAD.
            transcription_config=TranscriptionConfig(
                language=LANGUAGE, enable_partials=True, additional_vocab=CUSTOM_VOCAB
            ),
            record_events=False,
        )
        self.register_stt_handlers(self.stt)
        await self.stt.connect()  # returns once the service is ready for audio

    def register_stt_handlers(self, client: AgentSttAsyncClient) -> None:
        """SDK handlers are synchronous callbacks: anything slow is handed to a task."""

        @client.on(ServerMessageType.RECOGNITION_STARTED)
        def on_started(message):
            log("STT       Speechmatics session started", message.get("id"))

        # StartOfTurn fires on the caller's first recognized word: a new turn, so the
        # reply, spoken or not, is out of date and is dropped.
        @client.on(ServerMessageType.START_OF_TURN)
        def on_turn_start(_message):
            self.spawn(self.barge_in())

        # SpeechStarted fires on any voice activity, which can be a cough or line noise.
        # It pauses the agent at once, so barge-in stays quick, but keeps the reply.
        @client.on(ServerMessageType.SPEECH_STARTED)
        def on_voice_activity(_message):
            if self.speaking or self.paused:
                self.spawn(self.pause())

        @client.on(ServerMessageType.ADD_SEGMENT)
        def on_segment(message):
            text = message["segment"]["transcript"].strip()
            if text:
                self.turn_text.append(text)
                publish("PARTIAL", " ".join(self.turn_text))

        # EndOfTurn is the last message of a turn: the moment to answer.
        @client.on(ServerMessageType.END_OF_TURN)
        def on_turn_end(_message):
            text = " ".join(self.turn_text).strip()
            self.turn_text = []
            if text:
                self.turn_ended_at = time.time()
                log(f"CALLER    {text}")
                self.reply_task = self.spawn(self.reply(text))

        @client.on(ServerMessageType.ERROR)
        def on_error(message):
            log("STT       error:", message.get("reason", message))

    # ---------- LLM + Speechmatics TTS ----------
    async def llm_sentences(self):
        """Stream the LLM reply and yield it one sentence at a time."""
        # max_completion_tokens is the current name; reasoning models reject the older max_tokens.
        body = {"model": LLM_MODEL, "messages": self.history, "stream": True, "max_completion_tokens": 200}
        if LLM_TEMPERATURE and LLM_TEMPERATURE.lower() != "default":
            body["temperature"] = float(LLM_TEMPERATURE)
        if LLM_REASONING_EFFORT:
            body["reasoning_effort"] = LLM_REASONING_EFFORT
        buf = ""
        async with self.http.post(LLM_URL, json=body, headers={"Authorization": f"Bearer {LLM_API_KEY}"}) as r:
            if r.status != 200:
                log("LLM       error", r.status, (await r.text())[:300])
                return
            async for raw in r.content:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                if not chunk.get("choices"):
                    continue
                buf += chunk["choices"][0].get("delta", {}).get("content") or ""
                while m := SENTENCE_END.match(buf):
                    yield m.group(1).strip()
                    buf = buf[m.end() :]
        if buf.strip():
            yield buf.strip()

    async def tts_request(self, text: str) -> aiohttp.ClientResponse:
        """Start a streamed 16 kHz PCM synthesis with the configured provider."""
        if TTS_PROVIDER == "cartesia":
            return await self.http.post(
                "https://api.cartesia.ai/tts/bytes",
                headers={"Authorization": f"Bearer {CARTESIA_API_KEY}", "Cartesia-Version": CARTESIA_VERSION},
                json={
                    "model_id": CARTESIA_MODEL,
                    "transcript": text,
                    "voice": {"id": CARTESIA_VOICE_ID},
                    "language": LANGUAGE,
                    "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": RATE},
                },
            )
        if TTS_PROVIDER == "elevenlabs":
            return await self.http.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream",
                params={"output_format": f"pcm_{RATE}"},
                headers={"xi-api-key": ELEVENLABS_API_KEY},
                json={"text": text, "model_id": ELEVENLABS_MODEL},
            )
        return await self.tts.generate(text=text, voice=SM_VOICE, output_format=tts.OutputFormat.RAW_PCM_16000)

    async def speak(self, text: str) -> None:
        """Stream the TTS provider's 16 kHz PCM to the caller as it arrives, keeping a copy."""
        try:
            response = await self.tts_request(text)
        except (tts.AuthenticationError, tts.TransportError, tts.ConnectionError, aiohttp.ClientError) as e:
            log(f"TTS       {TTS_NAME} error: {e}")
            return
        sentence = Sentence(text)
        self.sentences.append(sentence)
        first = True
        carry = b""
        try:
            if response.status != 200:
                log(f"TTS       {TTS_NAME} error", response.status, (await response.text())[:300])
                return
            async for chunk in response.content.iter_chunked(CHUNK):
                if first and self.turn_ended_at:
                    elapsed_ms = (time.time() - self.turn_ended_at) * 1000
                    log(f"LATENCY   end of caller turn -> first agent audio: {elapsed_ms:.0f} ms")
                    self.turn_ended_at = 0.0
                first = False
                chunk = carry + chunk
                cut = len(chunk) - (len(chunk) % 2)  # send whole 16-bit samples only
                chunk, carry = chunk[:cut], chunk[cut:]
                sentence.pcm += chunk
                await self.flush()
        except aiohttp.ClientError as e:
            log(f"TTS       {TTS_NAME} stream interrupted: {e}")
        finally:
            sentence.complete = True
            response.release()

    # ---------- Gemini Live: LLM and voice in one connection ----------
    async def start_gemini(self) -> None:
        """Open the Gemini Live session for this call and start reading its replies."""
        from google import genai
        from google.genai import types

        self.gl_types = types
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), http_options={"api_version": "v1beta"})
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=self.prompt,
            output_audio_transcription=types.AudioTranscriptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=GEMINI_VOICE))
            ),
            tools=[types.Tool(function_declarations=[types.FunctionDeclaration(
                name="end_call", description="End the phone call. Call it right after saying goodbye.")])],
            context_window_compression=types.ContextWindowCompressionConfig(sliding_window=types.SlidingWindow()),
        )
        self.gl_cm = client.aio.live.connect(model=GEMINI_LIVE_MODEL, config=config)
        self.gl = await self.gl_cm.__aenter__()
        log(f"LLM       Gemini Live session started ({GEMINI_LIVE_MODEL}, voice {GEMINI_VOICE})")
        self.spawn(self.gemini_receiver())

    def gl_emit_sentences(self, final: bool = False) -> None:
        """Log each complete sentence of the reply transcript as it arrives."""
        while m := SENTENCE_END.match(self.gl_text):
            log(f"AGENT     {m.group(1).strip()}")
            self.gl_text = self.gl_text[m.end():]
        if final and self.gl_text.strip():
            log(f"AGENT     {self.gl_text.strip()}")
            self.gl_text = ""

    def gl_finish(self) -> None:
        """A generation is over (finished or interrupted)."""
        if self.gl_skip_ends:
            # The end of a reply the caller replaced: keep discarding until the new one starts.
            self.gl_skip_ends -= 1
            self.gl_drop = self.gl_skip_ends > 0
            return
        if not self.gl_drop:
            self.gl_emit_sentences(final=True)
        if self.gl_sentence:
            self.gl_sentence.complete = True
        self.gl_generating = False
        self.gl_drop = False
        self.gl_text = ""
        if self.gl_done:
            self.gl_done.set()

    async def gemini_receiver(self) -> None:
        """Route Gemini audio and transcript into the current reply, for the whole call."""
        while True:
            async for msg in self.gl.receive():
                if msg.tool_call:
                    for fc in msg.tool_call.function_calls or []:
                        if fc.name == "end_call":
                            log("CALL      agent ended the call after its goodbye")
                            self.hangup_after_speech = True
                            self.spawn(self.hangup_when_played())
                        await self.gl.send_tool_response(function_responses=[
                            self.gl_types.FunctionResponse(id=fc.id, name=fc.name, response={"ok": True})])
                    continue
                sc = msg.server_content
                if sc is None:
                    continue
                if sc.model_turn and not self.gl_drop:
                    for part in sc.model_turn.parts or []:
                        if part.inline_data and part.inline_data.data and self.gl_sentence is not None:
                            if self.turn_ended_at:
                                elapsed_ms = (time.time() - self.turn_ended_at) * 1000
                                log(f"LATENCY   end of caller turn -> first agent audio: {elapsed_ms:.0f} ms")
                                self.turn_ended_at = 0.0
                            data = self.gl_carry + part.inline_data.data
                            cut = len(data) - (len(data) % 2)
                            data, self.gl_carry = data[:cut], data[cut:]
                            self.gl_sentence.pcm += data
                            await self.flush()
                if sc.output_transcription and sc.output_transcription.text and not self.gl_drop:
                    self.gl_text += sc.output_transcription.text
                    if self.gl_sentence is not None:
                        self.gl_sentence.text += sc.output_transcription.text
                    self.gl_emit_sentences()
                if sc.interrupted or sc.generation_complete:
                    self.gl_finish()

    async def hangup_when_played(self) -> None:
        """Stop the stream once the goodbye has played (the tool call can come after it)."""
        await asyncio.sleep(0.3)
        while self.gl_generating or time.monotonic() < self.play_end + PLAYBACK_MARGIN_SECONDS:
            await asyncio.sleep(0.2)
        if self.hangup_after_speech:
            await self.send_vobiz({"event": "stop", "streamId": self.stream_id})

    async def gemini_turn(self, text: str) -> None:
        """Send one turn to Gemini Live and wait until its spoken reply has been generated."""
        self.start_speech()
        if self.gl_generating:
            # The previous reply is still being generated: this turn interrupts it, and its
            # remaining audio is discarded until Gemini reports that it ended.
            self.gl_skip_ends += 1
            self.gl_drop = True
        else:
            self.gl_drop = False
        self.gl_sentence = Sentence("")
        self.sentences.append(self.gl_sentence)
        self.gl_text, self.gl_carry = "", b""
        self.gl_done = asyncio.Event()
        self.gl_generating = True
        await self.gl.send_client_content(
            turns=self.gl_types.Content(role="user", parts=[self.gl_types.Part(text=text)]), turn_complete=True)
        try:
            await asyncio.wait_for(self.gl_done.wait(), timeout=20)
        except asyncio.TimeoutError:
            log("LLM       Gemini Live reply timed out")
            self.gl_skip_ends, self.gl_drop, self.gl_generating = 0, False, False
        self.speech_done = True
        await self.flush()

    async def reply(self, user_text: str) -> None:
        """Answer one caller turn, speaking each sentence as soon as the LLM writes it."""
        if GEMINI_LIVE:
            await self.gemini_turn(user_text)
            return
        self.history.append({"role": "user", "content": user_text})
        self.start_speech()
        said = []
        try:
            async for sentence in self.llm_sentences():
                end = HANGUP_TOKEN in sentence
                sentence = sentence.replace(HANGUP_TOKEN, "").strip()
                if sentence:
                    log(f"AGENT     {sentence}")
                    said.append(sentence)
                    await self.speak(sentence)
                if end:
                    self.hangup_after_speech = True
            self.speech_done = True
            await self.flush()  # sends the end-of-speech checkpoint, unless paused
        except asyncio.CancelledError:
            said.append("(interrupted)")
            raise
        finally:
            if said:
                self.history.append({"role": "assistant", "content": " ".join(said)})

    async def greet(self) -> None:
        """Speak the opening line."""
        if GEMINI_LIVE:
            await self.gemini_turn(f'(The phone call has just connected. Greet the caller with exactly: "{GREETING}")')
            return
        log(f"AGENT     {GREETING}")
        self.history.append({"role": "assistant", "content": GREETING})
        self.start_speech()
        await self.speak(GREETING)
        self.speech_done = True
        await self.flush()

    async def end_after(self, seconds: int) -> None:
        """Stop the stream after the time limit; with no XML after it, Vobiz hangs up."""
        await asyncio.sleep(seconds)
        log(f"CALL      reached the {seconds} s limit, ending the call")
        await self.send_vobiz({"event": "stop", "streamId": self.stream_id})

    # ---------- main loop ----------
    async def run(self) -> None:
        """Handle Vobiz stream events until the call ends."""
        try:
            async for msg in self.vobiz:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                event = data.get("event")
                if event == "media":
                    if self.stt:
                        await self.stt.send_audio(base64.b64decode(data["media"]["payload"]))
                elif event == "start":
                    if self.stt:
                        continue  # one agent STT session per stream
                    start = data["start"]
                    self.stream_id = start.get("streamId")
                    log(f"CALL      started call={start.get('callId')} format={start.get('mediaFormat')}")
                    try:
                        await self.start_stt()
                    except AuthenticationError:
                        log("STT       could not start a Speechmatics session: check SPEECHMATICS_API_KEY")
                        break
                    except Exception as e:  # noqa: BLE001 - network and session errors
                        log(f"STT       could not start a Speechmatics session: {e}")
                        break
                    if GEMINI_LIVE:
                        try:
                            await self.start_gemini()
                        except Exception as e:  # noqa: BLE001
                            log(f"LLM       could not start Gemini Live: {e}")
                            break
                    self.reply_task = self.spawn(self.greet())
                    self.spawn(self.end_after(MAX_CALL_SECONDS))
                elif event == "playedStream":
                    # Skip an answer to a checkpoint whose audio was cleared, or while paused.
                    name = data.get("name")
                    if self.paused or (name and name != self.checkpoint_name):
                        continue
                    self.speaking = False
                    if self.hangup_after_speech:
                        log("CALL      goodbye played, ending the call")
                        await self.send_vobiz({"event": "stop", "streamId": self.stream_id})
                elif event == "dtmf":
                    log(f"DTMF      caller pressed {data['dtmf']['digit']}")
        finally:
            # Vobiz sends no stop event: the socket closing is the end of the call.
            await self.close()

    async def close(self) -> None:
        """Cancel the call's tasks and close its agent STT session."""
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.gl_cm is not None:
            try:
                await asyncio.wait_for(self.gl_cm.__aexit__(None, None, None), timeout=5)
            except Exception:  # noqa: BLE001
                pass
        if self.stt:
            try:
                # Sends EndOfStream and waits briefly for the service to flush.
                await asyncio.wait_for(self.stt.disconnect(), timeout=5)
            except Exception:  # noqa: BLE001 - the call is over either way
                pass
        log("CALL      ended")


async def answer(_request: web.Request) -> web.Response:
    """Return the VobizXML that connects the call to the audio WebSocket."""
    log("HTTP      answer URL fetched")
    return web.Response(text=answer_xml(), content_type="application/xml")


async def hangup(request: web.Request) -> web.Response:
    """Log the hangup callback Vobiz sends when the call ends."""
    form = await request.post() if request.can_read_body else request.query
    log(f"HTTP      hangup cause={form.get('HangupCause')} duration={form.get('Duration')}")
    return web.Response(text="OK")


async def media_stream(request: web.Request) -> web.WebSocketResponse:
    """Accept the Vobiz audio WebSocket and run one call on it."""
    sock = web.WebSocketResponse()
    await sock.prepare(request)
    if request.app["calls"]["active"] >= MAX_CALLS:
        log(f"CALL      rejected: {MAX_CALLS} calls already in progress")
        await sock.close()
        return sock
    request.app["calls"]["active"] += 1
    try:
        await Call(sock, request.app).run()
    finally:
        request.app["calls"]["active"] -= 1
    return sock


async def live_page(_request: web.Request) -> web.Response:
    """The live call page: transcript, latency and pipeline, updated as the call runs."""
    html = (Path(__file__).with_name("live.html")).read_text()
    return web.Response(text=html, content_type="text/html")


async def live_events(request: web.Request) -> web.WebSocketResponse:
    """WebSocket for the live page: config first, then recent and new events.

    A WebSocket rather than server-sent events, so tunnels and proxies that buffer
    responses still deliver each event at once.
    """
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    config = {"kind": "CONFIG", "number": DEMO_NUMBER, "llm": GEMINI_LIVE_MODEL if GEMINI_LIVE else LLM_MODEL,
              "tts": TTS_NAME, "stt": "Speechmatics Agent STT", "stt_url": SM_RT_URL or "default (EU)"}
    queue: asyncio.Queue = asyncio.Queue()
    LISTENERS.add(queue)
    try:
        for event in [config, *EVENTS]:
            await ws.send_str(json.dumps(event))
        while not ws.closed:
            event = await queue.get()
            await ws.send_str(json.dumps(event))
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        LISTENERS.discard(queue)
    return ws


async def on_startup(app: web.Application) -> None:
    """Create the shared HTTP and TTS clients."""
    app["http"] = aiohttp.ClientSession()
    app["tts"] = tts.AsyncClient(api_key=SM_API_KEY)


async def on_cleanup(app: web.Application) -> None:
    """Close the shared HTTP and TTS clients."""
    await app["tts"].close()
    await app["http"].close()


def make_app() -> web.Application:
    """Build the web app with token-protected routes."""
    app = web.Application()
    app["prompt"] = system_prompt()
    app["calls"] = {"active": 0}  # mutable: app state is frozen once the server starts
    app.router.add_route("*", f"/answer/{PATH_TOKEN}", answer)
    app.router.add_route("*", f"/hangup/{PATH_TOKEN}", hangup)
    app.router.add_get(f"/ws/{PATH_TOKEN}", media_stream)
    app.router.add_get(f"/live/{PATH_TOKEN}", live_page)
    app.router.add_get(f"/events/{PATH_TOKEN}", live_events)
    app.router.add_static("/assets/", Path(__file__).resolve().parents[2] / "logo")
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def check_config() -> None:
    """Exit with a clear message when a required setting is missing."""
    missing = [
        name
        for name, value in (
            ("SPEECHMATICS_API_KEY", SM_API_KEY),
            ("GEMINI_API_KEY or LLM_API_KEY", os.getenv("GEMINI_API_KEY") if GEMINI_LIVE else LLM_API_KEY),
            ("PUBLIC_URL", PUBLIC_URL),
        )
        if not value
    ]
    if TTS_PROVIDER not in TTS_NAMES:
        sys.exit(f"TTS_PROVIDER must be one of: {', '.join(TTS_NAMES)} (got {TTS_PROVIDER!r})")
    if TTS_PROVIDER == "cartesia" and not CARTESIA_API_KEY:
        missing.append("CARTESIA_API_KEY (or set TTS_PROVIDER)")
    if TTS_PROVIDER == "elevenlabs" and not ELEVENLABS_API_KEY:
        missing.append("ELEVENLABS_API_KEY (or set TTS_PROVIDER)")
    if missing:
        sys.exit(f"Missing required settings in .env: {', '.join(missing)}")
    if not PUBLIC_URL.startswith("https://"):
        sys.exit(
            "PUBLIC_URL must start with https:// - Vobiz fetches the answer URL over HTTPS and streams over wss://"
        )


if __name__ == "__main__":
    check_config()
    log(f"Answer URL: {PUBLIC_URL}/answer/{PATH_TOKEN}")
    log(f"Hangup URL: {PUBLIC_URL}/hangup/{PATH_TOKEN}")
    log(f"Live page:  {PUBLIC_URL}/live/{PATH_TOKEN}")
    voice = {"cartesia": CARTESIA_VOICE_ID, "elevenlabs": ELEVENLABS_VOICE_ID, "gemini-live": GEMINI_VOICE}.get(
        TTS_PROVIDER, SM_VOICE.value)
    llm = GEMINI_LIVE_MODEL if GEMINI_LIVE else LLM_MODEL
    log(f"TTS: {TTS_NAME} ({voice})  LLM: {llm}  STT: {SM_RT_URL or 'default'}  Listening on http://127.0.0.1:{PORT}")
    web.run_app(make_app(), host="127.0.0.1", port=PORT, print=None)
