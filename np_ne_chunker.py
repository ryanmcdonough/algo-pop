import nltk, re, pprint, glob
from nltk import word_tokenize
from nltk.corpus import conll2000

class ConsecutiveNPChunkTagger(nltk.TaggerI):
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent =  nltk.tag.untag(tagged_sent) #[w for ((w, t), c) in tagged_sent]
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train(
            train_set, algorithm='megam', trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                        nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)

def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
    else:
        prevword, prevpos = sentence[i-1]
    if i == len(sentence)-1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i+1]
    return {"pos": pos,
            "word": word,
            "prevpos": prevpos,
            "nextpos": nextpos, 
            "prevpos+pos": "%s+%s" % (prevpos, pos), 
            "pos+nextpos": "%s+%s" % (pos, nextpos),
            "tags-since-dt": tags_since_dt(sentence, i)} 

def tags_since_dt(sentence, i):
    tags = set()
    for word, pos in sentence[:i]:
        if pos == 'DT':
            tags = set()
        else:
            tags.add(pos)
    return '+'.join(sorted(tags))

nounish = ["NN", "NNP", "NNPS", "NNS", "PRP", "PRP$"]
NP_tags = ["B-NP", "I-NP"]
named_entities = ["PERSON", "LOCATION", "DATE", "TIME", "ORGANIZATION", "MONEY", "PERCENT", "FACILITY", "GPE"]
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
print "Starting chunker training"
chunker = ConsecutiveNPChunker(train_sents)
print "Finished chunker training"
BOM = re.compile('\ufeff')
filelist = glob.glob("./template_treatments/*.txt")
for (n, txtfile) in enumerate(filelist):
    print "======= processing treatment: ", txtfile, " ======="
    f = open(txtfile)
    raw = f.read().decode('utf8')#.replace("\n", " ")
    words = word_tokenize(raw)
    tagged = nltk.pos_tag(words)
    
    #NAMED-ENTITY CHUNKED FILES:
    ne_chunks = nltk.ne_chunk(tagged)
    chunked_name = "ne_chunked_treatment_" + str(n) + ".nech"
    chunked_file = file("".join(chunked_name), 'w')
    chunked_file.write(str(ne_chunks))
    chunked_file.close()
    #Collect named entities across treatments
    collfile = file("ne_collection.nech", 'r')
    ne_collection = collfile.read().split("\n")
    collfile.close()
    collfile = file("ne_collection.nech", 'a')
    for subtree in ne_chunks.subtrees():
        if subtree.label() in named_entities:
            if(str(subtree)) not in ne_collection:
                print str(subtree)
                ne_collection.append(str(subtree))
                collfile.write("\n"+str(subtree))
    collfile.close()

    ## NOUN-PHRASE CHUNKED FILES:
    np_chunks = chunker.parse(tagged)
    chunked_name = "np_chunked_treatment_" + str(n) + ".npch"
    chunked_file = file("".join(chunked_name), 'w')
    chunked_file.write(str(np_chunks))
    chunked_file.close()
    #Collect noun-phrases across treatments
    collfile = file("np_collection.npch", 'r')
    np_collection = collfile.read().split("\n")
    collfile.close()
    collfile = file("np_collection.npch", 'a')
    for subtree in np_chunks.subtrees():
        if subtree.label() == "NP":
            if(str(subtree)) not in np_collection:
                print str(subtree)
                np_collection.append(str(subtree))
                collfile.write("\n"+str(subtree))
    collfile.close()
    f.close()
