import torchaudio as ta

# When installed, import directly from onelab
# For development without install, you might need: import sys; sys.path.append('src')
from onelab import ConversationInput, OneLab

# Initialize the library
onelab = OneLab(device="cpu", root_dir=".")
tts = onelab.tts


data: ConversationInput = {
    "segments": [
        {
            "voice": "charlie",
            "text": " What if time didn’t move the same way for everyone?",
        },
        {
            "voice": "emilia",
            "text": "Why do we assume that every clock in the universe ticks identically?",
        },
        {
            "voice": "charlie",
            "text": "Because it feels natural… until physics tells us it isn’t.",
        },
        {
            "voice": "emilia",
            "text": "So what actually happens to time when something starts moving very fast?",
        },
        {
            "voice": "charlie",
            "text": "It slows down—measurably, predictably, and for everyone watching from the outside.",
        },
        {
            "voice": "emilia",
            "text": "Does that mean two people can experience different amounts of time?",
        },
        {"voice": "charlie", "text": "Yes. And both experiences are equally real."},
        {
            "voice": "emilia",
            "text": "Then why don’t we notice this in everyday life?",
        },
        {
            "voice": "charlie",
            "text": "Because we don’t move fast enough—and gravity around us is too gentle.",
        },
        {
            "voice": "emilia",
            "text": "What happens when gravity is no longer gentle?",
        },
        {
            "voice": "charlie",
            "text": "Time slows down again—this time because space itself is curved.",
        },
        {
            "voice": "emilia",
            "text": "Curved space sounds abstract… what does that even mean?",
        },
        {
            "voice": "charlie",
            "text": "It means objects don’t fall because they’re pulled, but because they follow bent paths.",
        },
        {"voice": "emilia", "text": "So gravity isn’t really a force?"},
        {
            "voice": "charlie",
            "text": "Not in the way we once thought—it’s geometry doing the work.",
        },
        {
            "voice": "emilia",
            "text": "What happens to time near something extremely massive… like a black hole?",
        },
        {
            "voice": "charlie",
            "text": "For an outside observer, time there almost comes to a stop.",
        },
        {
            "voice": "emilia",
            "text": "Does that mean the universe doesn’t share a single ‘now’?",
        },
        {
            "voice": "charlie",
            "text": "Exactly. ‘Now’ depends on where you are and how you move.",
        },
        {
            "voice": "emilia",
            "text": "Then why does any of this matter to us?",
        },
        {
            "voice": "charlie",
            "text": "Because without accounting for relativity, modern technology would fail.",
        },
        {"voice": "emilia", "text": "Even something as ordinary as GPS?"},
        {
            "voice": "charlie",
            "text": "Especially GPS—it corrects for time differences every single day.",
        },
        {
            "voice": "emilia",
            "text": "So we’re using relativity without realizing it?",
        },
        {"voice": "charlie", "text": "All the time."},
        {
            "voice": "emilia",
            "text": "If time, space, and motion depend on perspective…",
        },
        {
            "voice": "charlie",
            "text": "Then maybe the real question is—what does reality mean for you?",
        },
    ]
}

# Create podcast using the library
wav = tts.podcast.create(data)
ta.save("conversation.wav", wav, tts.model.sr)
