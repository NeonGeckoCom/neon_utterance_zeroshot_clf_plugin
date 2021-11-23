```python
tars = TarsZeroShotClassifier()

# NOTE: this usually comes from config and is not set like this
tars.classifiers = {"sentiment": ["happy", "sad"]}

utts = ["I am so glad you liked it!",
        "I get really sad when nobody reviews my PRs"]

_, context = tars.transform(utts)

# context is a dict of clf_name: [prediction(utt) for utt in utterances]
print(context)
# {'zeroshot_classifier': 
# {'sentiment': [[('happy', 0.8667009472846985)], [('sad', 0.9921779036521912)]]}}
```