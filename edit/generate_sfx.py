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
    # --- extra life (screenplay beats without dialogue) ---
    "el_title_pop": dict(
        text="A soft, subtle bubbly UI pop for on-screen text appearing. Clean, short, friendly.",
        duration_seconds=0.5, prompt_influence=0.7),
    "el_grandma_phone": dict(
        text="An elderly woman lifting a smartphone and holding it up close to her face: light "
             "hand and fabric rustle, small plastic phone-case handling noise. Close-miked.",
        duration_seconds=1.2, prompt_influence=0.5),
    "el_grandma_home": dict(
        text="Cosy grandmother's kitchen ambience: a mantel clock ticking softly and a kettle "
             "quietly simmering on a stove, calm and warm, no voices.",
        duration_seconds=9.0, prompt_influence=0.5),
    "el_sparkle": dict(
        text="A cheesy cartoon sparkle twinkle with a bright 'ding', like a shiny gleam off "
             "perfectly styled hair.",
        duration_seconds=1.0, prompt_influence=0.7),
    "el_wind_mic": dict(
        text="Light wind buffeting a phone microphone while someone walks outdoors, soft gusts "
             "and mic rumble, no voices.",
        duration_seconds=12.0, prompt_influence=0.5),
    "el_car_pass": dict(
        text="A single car driving past on a city street, left to right, moderate speed.",
        duration_seconds=2.5, prompt_influence=0.6),
    "el_rimshot": dict(
        text="Cheesy comedy rimshot sting 'ba-dum-tss' on a drum kit with a cymbal splash.",
        duration_seconds=1.2, prompt_influence=0.8),
    "el_clock_tick": dict(
        text="A slow wall clock ticking loudly in a silent bedroom, one tick per second, "
             "nothing else.",
        duration_seconds=5.0, prompt_influence=0.7),
    "el_peanut_bag": dict(
        text="Hand rummaging in a crinkly plastic snack bag, then peanut shells cracking open "
             "close to the microphone.",
        duration_seconds=2.0, prompt_influence=0.6),
    "el_bed_creak": dict(
        text="A person shifting on a bed: a single mattress spring creak and a short blanket "
             "rustle. Close, indoor.",
        duration_seconds=1.0, prompt_influence=0.6),
    "el_nose_breathing": dict(
        text="Heavy nasal breathing very close to a phone microphone, slow loud breaths in and "
             "out through the nose, slightly gross, no voice.",
        duration_seconds=4.0, prompt_influence=0.6),
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
