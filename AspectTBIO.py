# coding: utf-8

# Convert Aspect Term to BIO format
import xml.etree.ElementTree as ET
from collections import defaultdict
import nltk
from sklearn.preprocessing import LabelBinarizer
from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix

def bio_classification_report(y_true, y_pred):
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_)  #- {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

  
def AspectTerm2BIO(sentence,Aspect_Term_List,sentence_id):
    sentence_tokens = nltk.word_tokenize(sentence)
    array_sentences = []
    word_pos = nltk.pos_tag(sentence_tokens) #part of speech
    offset = 0
    Aspect_Term_counter = 0 #Iterate over the aspect list
    for word in word_pos:
        BOI = "O"
        offset = sentence.find(word[0],offset)
        word_text = word[0]
        word_start = offset
        word_end = offset+len(word[0])
        word_part_of_speech = word[1]
        
        if Aspect_Term_counter < len(Aspect_Term_List):
            if (word_start == Aspect_Term_List[Aspect_Term_counter][1] ):
                BOI = "B"
            elif (word_start > Aspect_Term_List[Aspect_Term_counter][1]):
                BOI = "I"
            if word_end == Aspect_Term_List[Aspect_Term_counter][2]:
                Aspect_Term_counter += 1
        array_sentences.append((word[0],word[1],BOI))
        offset = word_end
    return array_sentences


def ParseXML(tree):  
    root = tree.getroot()
    d = {}
    aspect_Terms = defaultdict(list)
    count = 0
    for sentence in root:
        id = sentence.attrib.get("id")
        for text in sentence.iter("text"):
            text_value = text.text
            d[id] =text_value
        for aspectTerms in sentence.iter("aspectTerms"):
            for aspectTerm in aspectTerms.iter("aspectTerm"):
                from_value = int(aspectTerm.attrib.get("from"))
                term_value  = aspectTerm.attrib.get("term")
                to_value = int(aspectTerm.attrib.get("to"))
                aspect_Terms[id].append((term_value,from_value,to_value))

    for k,v in aspect_Terms.iteritems():
        v.sort(key=lambda x: x[1])
    dict_vector = {}
    sentence_array = []
    for k,v in d.iteritems():
        sentence_array.append(AspectTerm2BIO(v,aspect_Terms[k],k))

    return sentence_array
  
  
# lesser features testing the accuracy,precision,recall - Iteration 1
def add_sent_features(sent):
    return [add_word_features(sent, i) for i in range(len(sent))]
    
def add_sent_labels(sent):
    return [label for token, postag, label in sent]

def add_sent_tokens(sent):
    return [token for token, postag, label in sent]    

def add_word_features(sent,i):
    word = sent[i][0]
    postag = sent[i][1]
    if len(sent)> 10:
        sentence_length = "long"
    else:
        sentence_length = "short"
    features = [
        'word.lower=' + word.lower(),
        'sentence_length=%s'% sentence_length,
        'word.istitle=%s' % word.istitle(),
        'word.isupper=%s' % word.isupper(),
        'word.islower=%s' % word.islower(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'word.pos_in_sent=%s' % (i + 1),
        #'bias',
        #'word[-3:]=' + word[-3:],
        #'word[-2:]=' + word[-2:],
        #'postag[:2]=' + postag[:2],
    ]
    #print features
    return features


train_set = ParseXML(ET.parse('Data/Laptops_Train_v2.xml'))
test_set = ParseXML(ET.parse('Data/Laptops_Test_Gold.xml'))


X_train = [add_sent_features(s) for s in train_set]
y_train = [add_sent_labels(s) for s in train_set]

X_test = [add_sent_features(s) for s in test_set]
y_test = [add_sent_labels(s) for s in test_set]






# Algorithm 1: LBFGS
import pycrfsuite
trainer = pycrfsuite.Trainer(verbose=False)

trainer.set_params({
    'c1': 0.01,   # coefficient for L1 penalty
    'c2': 1,  # coefficient for L2 penalty
   'max_iterations': 100 

    # include transitions that are possible, but not observed
    #'feature.possible_transitions': True
})

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.train('archana_bfgs1.model')


tagger = pycrfsuite.Tagger()
tagger.open('archana_bfgs1.model')


#y_pred = [tagger.tag(xseq) for xseq in X_train]
y_pred = [tagger.tag(xseq) for xseq in X_test]

#print(bio_classification_report(y_train, y_pred))
print "Algorithm 1: LBFGS"
print(bio_classification_report(y_test, y_pred))


# Algorithm 2: l2sgd
trainer2 = pycrfsuite.Trainer(verbose=False)

trainer2.select("l2sgd")

for xseq, yseq in zip(X_train, y_train):
    trainer2.append(xseq, yseq)

trainer2.train('archana_l2sgd.model')

tagger = pycrfsuite.Tagger()
tagger.open('archana_l2sgd.model')

#y_pred = [tagger.tag(xseq) for xseq in X_train]
y_pred = [tagger.tag(xseq) for xseq in X_test]

#print(bio_classification_report(y_train, y_pred))
print "Algorithm 2: l2sgd"
print(bio_classification_report(y_test, y_pred))




# Algorithm 3: ap
trainer3 = pycrfsuite.Trainer(verbose=False)

trainer3.select("ap")



for xseq, yseq in zip(X_train, y_train):
    trainer3.append(xseq, yseq)

trainer3.train('archana_ap.model')

tagger = pycrfsuite.Tagger()
tagger.open('archana_ap.model')

#y_pred = [tagger.tag(xseq) for xseq in X_train]
y_pred = [tagger.tag(xseq) for xseq in X_test]

#print(bio_classification_report(y_train, y_pred))
print "Algorithm 3: ap"

print(bio_classification_report(y_test, y_pred))




# Algorithm 4: pa
trainer4 = pycrfsuite.Trainer(verbose=False)

trainer4.select("pa")

for xseq, yseq in zip(X_train, y_train):
    trainer4.append(xseq, yseq)

trainer4.train('archana_pa.model')

tagger = pycrfsuite.Tagger()
tagger.open('archana_pa.model')

#y_pred = [tagger.tag(xseq) for xseq in X_train]
y_pred = [tagger.tag(xseq) for xseq in X_test]

#print(bio_classification_report(y_train, y_pred))
print "Algorithm 4: pa"

print(bio_classification_report(y_test, y_pred))


# Algorithm 5: arow
trainer5 = pycrfsuite.Trainer(verbose=False)

trainer5.select("arow")

for xseq, yseq in zip(X_train, y_train):
    trainer5.append(xseq, yseq)

trainer5.train('archana_arow.model')

tagger = pycrfsuite.Tagger()
tagger.open('archana_arow.model')

#y_pred = [tagger.tag(xseq) for xseq in X_train]
y_pred = [tagger.tag(xseq) for xseq in X_test]

#print(bio_classification_report(y_train, y_pred))
print "Algorithm 5: arow"

print(bio_classification_report(y_test, y_pred))



