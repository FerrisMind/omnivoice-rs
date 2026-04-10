import argparse
import base64
import io
import wave

from openai import OpenAI


def make_ref_audio_data_uri() -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        frames = bytearray()
        for i in range(24000):
            sample = int(0.1 * 32767 * __import__("math").sin(i / 8.0))
            frames += sample.to_bytes(2, "little", signed=True)
        wav_file.writeframes(bytes(frames))
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test omnivoice-server via openai-python")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    wav_bytes = client.audio.speech.create(
        model="default",
        voice="alloy",
        input="FerrisMind server wav smoke test.",
        response_format="wav",
    ).content
    assert len(wav_bytes) > 44, "wav response is unexpectedly small"

    mp3_bytes = client.audio.speech.create(
        model="default",
        voice="alloy",
        input="FerrisMind server mp3 smoke test.",
        response_format="mp3",
    ).content
    assert len(mp3_bytes) > 0, "mp3 response is empty"

    clone_bytes = client.audio.speech.create(
        model="default",
        voice="alloy",
        input="FerrisMind clone path smoke test.",
        response_format="wav",
        extra_body={
            "language": "en",
            "ref_text": "reference text",
            "ref_audio": make_ref_audio_data_uri(),
            "instruct": "male",
        },
    ).content
    assert len(clone_bytes) > 44, "clone wav response is unexpectedly small"

    streamed_pcm = bytearray()
    with client.audio.speech.with_streaming_response.create(
        model="default",
        voice="alloy",
        input="FerrisMind streaming pcm smoke test.",
        response_format="pcm",
    ) as response:
        for chunk in response.iter_bytes():
            streamed_pcm.extend(chunk)
    assert len(streamed_pcm) > 0, "streamed pcm response is empty"

    print("wav_bytes", len(wav_bytes))
    print("mp3_bytes", len(mp3_bytes))
    print("clone_bytes", len(clone_bytes))
    print("streamed_pcm_bytes", len(streamed_pcm))


if __name__ == "__main__":
    main()
