# Sentence Boundary Disambiguation

A sample NLP model using ML to detect boundaries of a given sentence and to classify them.

The dataset used is the **Brown Corpus (POS-tagged)**. It can be found [here](https://www.kaggle.com/nltkdata/brown-corpus).

This paper was used for reference - 

```Agarwal, Neha, Kelley Herndon Ford, and Max Shneider. "Sentence boundary detection using a maxEnt classifier." Proceedings of MISC. 2005.```


## Methodology -

1. Sample sentences of the dataset looks like this -

`'\tIn/in a/at few/ap school/nn districts/nns one/pn finds/vbz a/at link/nn between/in school/nn and/cc job/nn ./.\n',
'In/in those/dts vocational/jj programs/nns organized/vbn with/in Smith-Hughes/np money/nn ,/, there/ex may/md be/be a/at close/jj tie/nn between/in the/at labor/nn union/nn and/cc a/at local/jj employer/nn on/in the/at one/cd hand/nn and/cc the/at vocational/jj teacher/nn on/in the/at other/ap ./.\n',`

First we clean the data by removing escape characters (like ‘/n’,’/t’, e.t.c.) and strip the extra white spaces. Then we join 2 sentences together so that we can train the model to detect boundaries in a paragraph and not just a single sentence. 

The POS-tag for boundary is ‘/.’ so if a word has ‘/.’ as its tag we append an additional label ‘/Y’ else we label it ‘/N’.

So the output sentence finally looks like this -

`'In/in/N a/at/N few/ap/N school/nn/N districts/nns/N one/pn/N finds/vbz/N a/at/N link/nn/N between/in/N school/nn/N and/cc/N job/nn/N ././Y In/in/N those/dts/N vocational/jj/N programs/nns/N organized/vbn/N with/in/N Smith-Hughes/np/N money/nn/N ,/,/N there/ex/N may/md/N be/be/N a/at/N close/jj/N tie/nn/N between/in/N the/at/N labor/nn/N union/nn/N and/cc/N a/at/N local/jj/N employer/nn/N on/in/N the/at/N one/cd/N hand/nn/N and/cc/N the/at/N vocational/jj/N teacher/nn/N on/in/N the/at/N other/ap/N ././Y’`

2. For each word, we add 4 features, those are -
* Check if words contains ".","?","!",";" .
* Check if the word is at the end of the sentence .
* Check for initials or abbreviations, eg. "A. B." or "e.g."
* Check if word next to ".",";","?","!" is in uppercase(the first letter) and this word is in lowercase(the first letter).

The results look something like this -

![dataframe](https://user-images.githubusercontent.com/59285634/132118038-3e93bf4a-2734-4181-a06c-a72b08adfae8.png)

3. We first use stratified k folds to split the dataset into training and testing parts, because the ratio of Y:N classes is approximately 5:100, so we face a clear class imbalance.
The total records that we train the model on is about 600,000.
Then we train a Random Forest Classifier with 10 trees and max depth of 5 so as to prevent overfitting (if any occurs).

## Accuracy

The accuracy obtained for training data is 99.52% and for test data is 99.59%.

## Sample Test

For the sake of evaluating the model on real life sentences, I took a random paragraph from wikipedia and predicted boundaries in those. The sentence was -

```'In common parlance, randomness is the apparent or actual lack of pattern or predictability in events . A random sequence of events, symbols or steps often has no order and does not follow an intelligible pattern or combination . Individual random events are, by definition, unpredictable, but if the probability distribution is known, the frequency of different outcomes over repeated events (or "trials") is predictable. For example, when throwing two dice, the outcome of any particular roll is unpredictable, but a sum of 7 will tend to occur twice as often as 4. In this view, randomness is not haphazardness ; it is a measure of uncertainty of an outcome. Randomness applies to concepts of chance, probability, and information entropy .'```

The output is -

```In/N common/N parlance,/N randomness/N is/N the/N apparent/N or/N actual/N lack/N of/N pattern/N or/N predictability/N in/N events/N ./Y A/N random/N sequence/N of/N events,/N symbols/N or/N steps/N often/N has/N no/N order/N and/N does/N not/N follow/N an/N intelligible/N pattern/N or/N combination/N ./Y Individual/N random/N events/N are,/N by/N definition,/N unpredictable,/N but/N if/N the/N probability/N distribution/N is/N known,/N the/N frequency/N of/N different/N outcomes/N over/N repeated/N events/N (or/N "trials")/N is/N predictable./Y For/N example,/N when/N throwing/N two/N dice,/N the/N outcome/N of/N any/N particular/N roll/N is/N unpredictable,/N but/N a/N sum/N of/N 7/N will/N tend/N to/N occur/N twice/N as/N often/N as/N 4./Y In/N this/N view,/N randomness/N is/N not/N haphazardness/N ;/Y it/N is/N a/N measure/N of/N uncertainty/N of/N an/N outcome./Y Randomness/N applies/N to/N concepts/N of/N chance,/N probability,/N and/N information/N entropy/N ./Y```
