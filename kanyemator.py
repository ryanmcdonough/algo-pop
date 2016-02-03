import nltk, re, pprint, glob, random, codecs
from nltk import word_tokenize

filelist = glob.glob("./np_chunks/*.npch")
collfile = file("np_collection.npch", 'r')
np_collection = collfile.read().split("~")
collfile.close()
collfile = file("ne_collection.nech", 'r')
ne_collection = collfile.read().split("~")
collfile.close()
for (n, txtfile) in enumerate(filelist):
    print "===== processing chunk file: ", txtfile, " ====="
    np_chunked = file(txtfile, 'r')
    raw = np_chunked.read()
    np_chunked.close()
    context = nltk.Tree.fromstring(raw)

    #Construct "np_set", a list of tuples formatted:
    #(noun-phrase, list-of-tags, list-of-words),
    #(Excluding any noun-phrases consisting of only WDT or PRP tags)
    #found in the current treatment file (context)
    np_set = []
    for subtree in context.subtrees(lambda t: t.height() == 2):
        if subtree.label() == "NP":
            # get tags. WDT and PRP, and containing POS at start
            # will need special treatment
            tags = []
            words = []
            for leaf in subtree.leaves():
                tags.append(nltk.tag.str2tuple(leaf)[1])
                words.append(nltk.tag.str2tuple(leaf)[0])
            if tags == ["WDT"] or tags == ["PRP"]:
                pass
            else:
                if subtree not in np_set:
                    np_set.append((str(subtree), tags, words))

    #Construct a dictionary "subs_dict",of {noun-phrase: substitution-np} 
    #pairs for each noun-phrase found in the context
    subs_dict = {}
    for np in np_set:
        #First check if current np is a named entity, and what type:
        isnp = False
        ne_coll_trees = [nltk.Tree.fromstring(string) for string in ne_collection]
        for tree in ne_coll_trees:
            leaves = tree.leaves()
            words = [nltk.tag.str2tuple(leaf)[0] for leaf in leaves]
            if words == np[2]:
                ne_tag = tree.label()
                isnp = True
                break
        if isnp:
            #pick randomly from a list of matching ne_tagged phrases:
            subs = random.choice([str(tree) for tree in ne_coll_trees if tree.label() == ne_tag])
            subs_dict[np[0]] = subs
            continue

        #otherwise find all noun-phrases with identical tag orders
        matches = []
        np_coll_trees = [nltk.Tree.fromstring(string) for string in np_collection]
        for tree in np_coll_trees:
            tags = []
            for leaf in tree.leaves():
                tags.append(nltk.tag.str2tuple(leaf)[1])
            if tags == np[1]:
                matches.append(str(tree))
        #if we find matching noun-phrases, choose one at random for substitution
        if len(matches) > 1:
            subs = random.choice(matches)
        else:
            subs = np[0]
        subs_dict[np[0]] = subs

    context_str = str(context).split("\n")
    replaced = []
    for string in context_str:
        stripped = string.strip()
        if stripped in dict.keys(subs_dict):
            replaced.append(subs_dict[stripped])
        else:
            replaced.append(stripped)
    replaced_str = " ".join(replaced)
    replacedTree = nltk.Tree.fromstring(replaced_str)
    newtext = []
    for leaf in replacedTree.leaves():
        newtext.append(nltk.tag.str2tuple(str(leaf))[0])
    newname = "generated_" + str(n) + ".txt"
    newfile = file(newname, 'w')
    newfile.write(" ".join(newtext))
    newfile.close()
                
