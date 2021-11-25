# # NEON AI (TM) SOFTWARE, Software Development Kit & Application Development System
# # All trademark and other rights reserved by their respective owners
# # Copyright 2008-2021 Neongecko.com Inc.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS;  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from flair.data import Sentence
from flair.models import TARSClassifier
from neon_transformers import UtteranceTransformer
from neon_transformers.tasks import UtteranceTask


class TarsZeroShotClassifier(UtteranceTransformer):
    task = UtteranceTask.ADD_CONTEXT

    def __init__(self, name="TarsZeroShotClassifier", priority=99):
        super().__init__(name, priority)
        # Load pre-trained TARS model for English
        self.tars = TARSClassifier.load('tars-base')
        self.classifiers = self.config.get("classifiers") or {}

        # dict of clf_name: [labels]
        # self.classifiers = {"sentiment": ["happy", "sad"]}

    def transform(self, utterances, context=None):
        preds = {}
        for clf, labels in self.classifiers.items():
            preds[clf] = []
            for utt in utterances:
                sentence = Sentence(utt)
                self.tars.predict_zero_shot(sentence, labels)
                preds[clf].append([(l.value, l.score)
                                    for l in sentence.labels])

        # return unchanged utterances + data
        return utterances, {"zeroshot_classifier": preds}


