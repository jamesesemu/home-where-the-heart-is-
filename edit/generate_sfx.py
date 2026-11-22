"""Generate the ElevenLabs sound effects used in the mix into assets/.

Usage: python3 generate_sfx.py [name ...]   (no names = generate all)

The SDK reads ELEVENLABS_API_KEY from the environment; the key is never
stored in code. Re-running overwrites the files, so only regenerate a sound
you want a new take of.
"""
import os
import sys

from elevenlabs import ElevenLabs

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

SOUNDS = {
    "el_phone_drop": dict(
        text="A smartphone slips and smacks face-down onto a wooden desk: one sharp "
             "plastic-and-glass clack, then a short skid and rattle. Close-miked, dry room.",
        duration_seconds=1.5, prompt_influence=0.6),
    "el_phone_fumble": dict(
        text="Frantic phone handling chaos: a smartphone repeatedly bangs, clatters and "
             "flips against a wooden desk while hands grab and fumble at it, quick thuds, "
             "slides and knocks. Close-miked, dry bedroom.",
        duration_seconds=3.0, prompt_influence=0.5),
    "el_phone_propup": dict(
        text="A smartphone is set upright against an object on a wooden desk: a light "
             "plastic tap and a tiny slide. Close-miked, dry room.",
        duration_seconds=0.8, prompt_influence=0.6),
    "el_tv_dialogue": dict(
        text="A TV sitcom playing across a bedroom: indistinct muffled dialogue between a "
             "man and a woman with a faint studio laugh track, heard from a distance through "
             "the TV speakers.",
        duration_seconds=10.0, prompt_influence=0.5),
}


def main(names):
    client = ElevenLabs()
    for name in names or SOUNDS:
        audio = client.text_to_sound_effects.convert(
            model_id="eleven_text_to_sound_v3", output_format="mp3_44100_192", **SOUNDS[name])
        path = os.path.join(ASSETS, f"{name}.mp3")
        with open(path, "wb") as f:
            for chunk in audio:
                f.write(chunk)
        print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1:])
