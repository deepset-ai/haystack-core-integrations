# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Union, Optional, List, Dict, Tuple, Any

from pathlib import Path

from torch.cuda import is_available as is_cuda_available
from tqdm.auto import tqdm
from haystack.nodes import BaseComponent
from haystack import Answer

from text2speech.utils import TextToSpeech


class AnswerToSpeech(BaseComponent):
    """
    This node adds an audio version of the answer and context into the `audio` metadata field of text-based Answers,
    plus some other additional information like the audio's sample rate.
    """

    outgoing_edges = 1

    def __init__(
        self,
        model_name_or_path: Union[str, Path] = "espnet/kan-bayashi_ljspeech_vits",
        generated_audio_dir: Path = Path("./generated_audio_answers"),
        audio_params: Optional[Dict[str, Any]] = None,
        transformers_params: Optional[Dict[str, Any]] = None,
        progress_bar: bool = True,
        devices: Optional[List[str]] = None,
    ):
        """
        Convert an input Answer into an audio file containing the answer and its context read out loud.

        :param model_name_or_path: The text-to-speech model, for example `espnet/kan-bayashi_ljspeech_vits`.
        :param generated_audio_dir: The folder to save the audio file to.
        :param audio_params: Additional parameters for the audio file. See `TextToSpeech` for details.
            The allowed parameters are:
            - audio_format: The format to save the audio into (wav, mp3, ...). Defaults to `wav`.
                Supported formats:
                - Uncompressed formats thanks to `soundfile` (see `libsndfile documentation
                    <https://libsndfile.github.io/libsndfile/api.html>`_ for a list of supported formats).
                - Compressed formats thanks to `pydub`
                    (uses FFMPEG: run `ffmpeg -formats` in your terminal to see the list of supported formats).
            - subtype: Used only for uncompressed formats. See `libsndfile documentation
                <https://libsndfile.github.io/libsndfile/api.html>`_ for the complete list of available subtypes.
                Defaults to `PCM_16`.
            - sample_width: Used only for compressed formats. The sample width of your audio. Defaults to 2.
            - channels count: Used only for compressed formats. The number of channels your audio file has:
                1 for mono, 2 for stereo. Depends on the model, but it's often mono so it defaults to 1.
            - bitrate: Used only for compressed formats. The desired bitrate of your compressed audio.
                Defaults to '320k'.
            - normalized: Used only for compressed formats. Normalizes the audio before compression (range 2^15)
                or leaves it untouched.
            - audio_naming_function: A function mapping the input text into the audio file name.
                By default, the audio file gets the name from the MD5 sum of the input text.
        :param transformers_params: The parameters to pass over to the `Text2Speech.from_pretrained()` call.
        :param progress_bar: Whether to show a progress bar while converting the text to audio.
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]).
                        NOTE: multiple devices are not supported yet. If many devices are passed,
                        only the first one will be used.
        """
        super().__init__()
        self.converter = TextToSpeech(
            model_name_or_path=model_name_or_path,
            transformers_params=transformers_params,
            devices=devices or ["cuda" if is_cuda_available() else "cpu"],
        )
        self.generated_audio_dir = generated_audio_dir
        self.params: Dict[str, Any] = audio_params or {}
        self.progress_bar = progress_bar

    def run(self, answers: List[Answer]) -> Tuple[Dict[str, List[Answer]], str]:  # type: ignore  # pylint: disable=arguments-differ
        for answer in tqdm(
            answers, disable=not self.progress_bar, desc="Converting answers to audio"
        ):

            if isinstance(answer.answer, str):
                answer_audio = self.converter.text_to_audio_file(
                    text=answer.answer,
                    generated_audio_dir=self.generated_audio_dir,
                    **self.params
                )
                answer.meta["audio"] = {
                    "answer": {
                        "path": answer_audio,
                        "format": self.params.get(
                            "audio_format", answer_audio.suffix.replace(".", "")
                        ),
                        "sample_rate": self.converter.model.fs,
                    }
                }
            if isinstance(answer.context, str):
                context_audio = self.converter.text_to_audio_file(
                    text=answer.context,
                    generated_audio_dir=self.generated_audio_dir,
                    **self.params
                )
                answer.meta["audio"]["context"] = {
                    "path": context_audio,
                    "format": self.params.get(
                        "audio_format", context_audio.suffix.replace(".", "")
                    ),
                    "sample_rate": self.converter.model.fs,
                }
        return {"answers": answers}, "output_1"

    def run_batch(self, answers: List[List[Answer]]) -> Tuple[Dict[str, List[List[Answer]]], str]:  # type: ignore  # pylint: disable=arguments-differ
        results: Dict[str, List[List[Answer]]] = {"answers": []}
        for answers_list in answers:
            results["answers"].append(self.run(answers_list)[0]["answers"])

        return results, "output_1"
