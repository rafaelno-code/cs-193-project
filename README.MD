# Hum-to-MIDI

## Problem
Songwriters often come up with melodies away from their instruments, but recording a voice memo leaves them with audio that has to be transcribed by ear later.

## Solution
I'm building an app that listens to my hummed or whistled melodies and turns them into clean MIDI I can drop straight into my DAW. My tool tracks the pitch of my voice, splits it into individual notes, and snaps them to the right key and beat, so my ideas go from my head to my project in seconds instead of getting lost in old voice memos.

## Core Libraries
- **torchcrepe**: detects the pitch of the hummed melody frame by frame
- **librosa**: loads the audio and estimates tempo for the beat grid
- **NumPy**: smooths the pitch curve and splits it into individual notes
- **pretty_midi**: writes the quantized notes out as a MIDI file

## Eisenhower Matrix

|                   | **Urgent** | **Not Urgent** |
|-------------------|------------|----------------|
| **Important**     | **Do first**<br>• Load WAV and run pitch detection<br>• Segment the pitch curve into notes<br>• Export notes to a MIDI file (first end-to-end result) | **Schedule**<br>• Estimate tempo and quantize timings to a beat grid<br>• Snap notes to a detected key<br>• Tune thresholds (confidence, minimum note length) against test hums<br>• Write tests for note segmentation and quantization |
| **Not Important** | **Do quickly / keep minimal**<br>• Set up virtual environment and dependencies<br>• Record a handful of sample hums to test with<br>• Basic repo setup (.gitignore, folders) | **Drop for now**<br>• Live mic recording<br>• Web or desktop UI<br>• In-app MIDI playback<br>• Comparing against basic-pitch<br>• Docker and deployment |

**Focus:** finish the top-left quadrant first. A rough WAV-to-MIDI result beats a polished setup with no output.
