from flask import Flask
from flask import render_template, request, redirect, url_for

import nltk, glob, random, os, codecs

mode = "SIMPLE" #"JUST_LEAD" #"SIMPLE" "MAX"
collfile = file("np_collection.npch", 'r')
np_collection = collfile.readlines()
collfile.close()
collfile = file("ne_collection.nech", 'r')
ne_collection = collfile.readlines()
collfile.close()

lead = "our protagonist"
location = "your home-town"

app = Flask(__name__)

@app.route("/")
def index():
    video = {}
    video['lead'] = "Click the pickle button..."
    video['text'] = "... make a tasty pickle"

    return render_template('the_pickler.html', video=video)

@app.route("/pickle", methods=['POST'])
def pickle():
    video = {}
    if mode == "JUST_LEAD":
        filelist = glob.glob("./template_treatments/originals/*.txt")
    elif mode == "SIMPLE":
        filelist = glob.glob("./template_treatments/noun_phrases/*.npch")
    else:
        filelist = glob.glob("./template_treatments/noun_phrases_long/*.npch")

    template_file = random.choice(filelist)
    np_chunked = codecs.open(template_file, encoding='utf-8', mode='r')
    np_chunked = file(template_file, 'r')
#    np_chunked = file(template_file, 'r')

    raw = np_chunked.read()
    np_chunked.close()
    
    if mode == "JUST_LEAD":
        subs = raw.replace("LEAD", lead).replace("\n", "<br>")
        video['title'] = request.form['title']
        video['text'] = subs
        return render_template('the_pickler.html', video=video)

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
    lead = request.form['lead']
    if lead == "":
        ne = random.choice([nltk.Tree.fromstring(ne) for ne in ne_collection if nltk.Tree.fromstring(ne).label() == "PERSON"])
        lead = " ".join([nltk.tag.str2tuple(leaf)[0] for leaf in ne.leaves()])

    lead = "<em>" + lead + "</em>"
    newtext = " ".join(newtext).replace("LEAD", lead)
    newtext = newtext.replace(" ,",",").replace(" .",".")
    video['lead'] = lead
    video['text'] = newtext
    
    return render_template('the_pickler.html', video=video)

if __name__ == "__main__":
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

