"""
This file is for loading MIDI and it replicates madmom.io.midi just to reduce dependency.
"""

import mido
import warnings
import jams

DEFAULT_TEMPO = 500000 # 500,000 microseconds per beat
DEFAULT_TICKS_PER_BEAT = 480
DEFAULT_TIME_SIGNATURE = (4, 4)

def tick2second(tick, ticks_per_beat=DEFAULT_TICKS_PER_BEAT,
                tempo=DEFAULT_TEMPO):
    # Note: both tempo (microseconds) and ticks are per quarter note
    #       thus the time signature is irrelevant
    # https://mido.readthedocs.io/en/latest/midi_files.html#tempo-and-beat-resolution
    scale = tempo*1e-6/ticks_per_beat #  = (sec/ticks)

    return tick*scale

def tick2beat(tick, ticks_per_beat=DEFAULT_TICKS_PER_BEAT,
              time_signature=DEFAULT_TIME_SIGNATURE):
    return tick/(4.0*ticks_per_beat/time_signature[1])

def note_hash(channel, pitch):
    return channel*128 + pitch

class MIDIFile(mido.MidiFile):
    def __init__(self, filename=None, file_format=0, ticks_per_beat=DEFAULT_TICKS_PER_BEAT, 
                 unit='seconds', timing='absolute', **kwargs):
        super(MIDIFile, self).__init__(filename=filename, type=file_format,
                                       ticks_per_beat=ticks_per_beat, **kwargs)
        self.unit = unit
        self.timing = timing

    def __iter__(self):
        if self.type == 2:
            raise TypeError("can't merge tracks in type 2 (asynchronous) file")

        tempo = DEFAULT_TEMPO
        time_signature = DEFAULT_TIME_SIGNATURE
        cum_delta = 0
        for msg in mido.merge_tracks(self.tracks):
            # Convert relative message time to desired unit
            if msg.time > 0:
                if self.unit.lower() in ('t', 'ticks'):
                    delta = msg.time
                elif self.unit.lower() in ('s', 'sec', 'seconds'):
                    delta = tick2second(msg.time, self.ticks_per_beat, tempo)
                elif self.unit.lower() in ('b', 'beats'):
                    delta = tick2beat(msg.time, self.ticks_per_beat,
                                      time_signature)
                else:
                    raise ValueError("`unit` must be either 'ticks', 't', "
                                     "'seconds', 's', 'beats', 'b', not %s." %
                                     self.unit)
            else:
                delta = 0
            # Convert relative time to absolute values if needed
            if self.timing.lower() in ('a', 'abs', 'absolute'):
                cum_delta += delta
            elif self.timing.lower() in ('r', 'rel', 'relative'):
                cum_delta = delta
            else:
                raise ValueError("`timing` must be either 'relative', 'rel', "
                                 "'r', or 'absolute', 'abs', 'a', not %s." %
                                 self.timing)

            yield msg.copy(time=cum_delta)

            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'time_signature':
                time_signature = (msg.numerator, msg.denominator)

def load_jamsfile(filename):
    '''
    Obtain a list object that contains ['onset','duration','note','channel','confidence'] value
    '''
    data = jams.load(filename)

    midi_data = data['annotations']['note_midi']
    notes_6 = [] # For 6 guitar strings.
    for guitar_string_num in range(6):
        sortedkeylist = midi_data[guitar_string_num].data
        notes = []
        for observation in sortedkeylist:
            notes.append([observation.time, 
                        observation.duration, round(observation.value), 0,
                        observation.confidence]) # MIDI value가 소수점까지 있는데, 우선 round(반올림) 하자...?
        sorted(notes) # <- 할 필요 없지 않나? 알아서 잘 정렬될듯. 그리고 안의 값들도 value가 아니라 list라 정렬안될듯.
        notes_6.append(notes)
    
    return notes_6