import json
import logging
import wave

from vosk import Model, KaldiRecognizer


class SpeechRecognizer:
    """Класс для распознавания речи"""

    def __init__(self, model_path):
        self.model = Model(model_path)

    def transcribe_audio(self, file_path):
        """Транскрипция аудио в текст"""
        with wave.open(file_path, "rb") as wf:
            rec = KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(True)

            logging.info("Starting transcription...")
            result_text = ""

            while True:
                data = wf.readframes(1024)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)

            result = rec.FinalResult()
            result_dict = json.loads(result)
            result_text = result_dict.get('text', '')
        return result_text
