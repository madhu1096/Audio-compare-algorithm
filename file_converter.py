from os import path
from pydub import AudioSegment

# files                                                                         
src = "/content/drive/My Drive/13 Sep, 4.22 PM.mp3"
dst = "myaudio3.wav"

# convert wav to mp3                                                            
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")