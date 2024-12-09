# Basic libraries
import os
import re
import time
import pandas as pd
# from collections import Counter

# NLTK
import nltk
from nltk.corpus import brown, gutenberg, webtext, inaugural, reuters, movie_reviews

# Stanza
import stanza

# Google Colab
from google.colab import drive

# NLTK downloads
nltk.download('brown')
nltk.download('gutenberg')
nltk.download('webtext')
nltk.download('inaugural')
nltk.download('reuters')
nltk.download('movie_reviews')
nltk.download('averaged_perceptron_tagger')

# Stanza download
stanza.download('en')

# Mount Google Drive
drive.mount('/content/drive')

# Example usage of corpora
corpora = {
    "Brown": brown.sents(),
    "Gutenberg": gutenberg.sents('austen-emma.txt'),
    "Webtext": webtext.sents('firefox.txt'),
    "Inaugural Address": inaugural.sents(),
    "Reuters": reuters.sents(),
    "Movie Reviews": movie_reviews.sents()
}

for corpus_name, sentences in corpora.items():
    print(f"{corpus_name} Corpus example sentence:", sentences[0])

# nltk.download('conll2000')
# from nltk.corpus import conll2000
# conll2000_sentences = conll2000.sents()
# print("Conll2000 Corpus example sentence:", conll2000_sentences[0])

# """# Standfor CoreNLP"""

# # install Java
# !apt-get install -y openjdk-11-jdk

# # download and unzip Stanford CoreNLP
# !wget http://nlp.stanford.edu/software/stanford-corenlp-4.4.0.zip
# !unzip stanford-corenlp-4.4.0.zip

!pip install stanza

import stanza

# download model
stanza.download('en')

import os
# from subprocess import Popen, PIPE

# load stanza pipeline
nlp = stanza.Pipeline('en', processors='tokenize,lemma,pos,depparse')

# # set CoreNLP path
# path_to_corenlp = "/content/stanford-corenlp-4.4.0"

# # activate CoreNLP server
# corenlp_server = Popen(['java', '-mx4g', '-cp', f'{path_to_corenlp}/*', 'edu.stanford.nlp.pipeline.StanfordCoreNLPServer'],
#                         stdout=PIPE, stderr=PIPE)

# # processing
# doc = nlp("Stanford University is located in California. It is a great university.")

# # output
# for sentence in doc.sentences:
#     for word in sentence.words:
#         print(f'text: {word.text}\tlemma: {word.lemma}\tpos: {word.pos}\thead: {word.head}\tdeprel: {word.deprel}')

# corenlp_server.terminate()

"""# Brown"""

# Define superlative patterns
superlative_patterns = [
    (re.compile(r'\b\w+est\b'), 'ADJ'),
    (re.compile(r'\b\w+iest\b'), 'ADJ'),
    (re.compile(r'\b(?:most|least|best|worst)\s+\w+\b'), 'ADJ'),
    (re.compile(r'\b(?:most|least|best|worst)\s+\w+ly\b'), 'ADV'),
    (re.compile(r'\b(?:foremost|hindmost|inmost|innermost|nethermost|outmost|outermost|topmost|undermost|upmost|uppermost|utmost|uttermost)\b'), 'ADJ'),
    (re.compile(r'\b(?:ablest|angriest|baldest|battiest|beadiest|bitterest|blackest|blandest|blankest|bleakest|blondest|bloodiest|bluest|bluntest|blurriest|boldest|bossiest|bounciest|brainiest|brashest|brassiest|bravest|brawniest|breeziest|briefest|brightest|briskest|broadest|brownest|bubbliest|bulkiest|bumpiest|burliest|bushiest|busiest|calmest|chattiest|cheapest|cheekiest|cheeriest|chewiest|chilliest|choicest|choppiest|chubbiest|chunkiest|clammiest|classiest|cleanest|clearest|cleverest|closest|cloudiest|clumsiest|coarsest|coldest|coolest|corniest|coziest|crabbiest|craftiest|crankiest|craziest|creakiest|creamiest|creepiest|crispest|crispiest|crudest|cruelest|crumbliest|crunchiest|crustiest|cuddliest|curliest|curviest|cutest|daffiest|daintiest|dampest|dandiest|darkest|deadliest|deepest|densest|dingiest|dirtiest|dizziest|dreamiest|dreariest|dressiest|driest|droopiest|drowsiest|dullest|dumbest|dustiest|earliest|easiest|edgiest|eeriest|emptiest|evilest|faintest|fairest|falsest|fanciest|fattest|faultiest|feeblest|fewest|ficklest|fiercest|fieriest|filmiest|filthiest|finest|firmest|fittest|flabbiest|flakiest|flashiest|flattest|flimsiest|floppiest|floweriest|fluffiest|foamiest|foggiest|fondest|foolhardiest|frailest|frankest|freakiest|freest|freshest|friendliest|frilliest|friskiest|frostiest|frothiest|fruitiest|frumpiest|fullest|funniest|furriest|fussiest|fuzziest|gabbiest|gaudiest|gauntest|gawkiest|gentlest|ghastliest|giddiest|glassiest|gloomiest|glossiest|goofiest|grainiest|grandest|gravest|greasiest|greatest|greediest|greenest|grimiest|grittiest|groggiest|grossest|grouchiest|grubbiest|gruffest|grumpiest|guiltiest|gustiest|gutsiest|hairiest|handiest|handsomest|happiest|hardest|hardiest|harshest|hastiest|haughtiest|haziest|healthiest|heartiest|heaviest|heftiest|highest|hippest|hoarsest|hollowest|homeliest|hottest|hugest|humblest|hungriest|huskiest|iciest|ickiest|itchiest|itty-bittiest|jazziest|jerkiest|jolliest|juiciest|kindest|kindliest|kingliest|knobbiest|knottiest|laciest|largest|laziest|leanest|lengthiest|lightest|likeliest|littlest|liveliest|loneliest|longest|loosest|loudest|lousiest|loveliest|lowest|lowliest|luckiest|lumpiest|maddest|meanest|meekest|mellowest|merriest|messiest|mightiest|mildest|mistiest|moistest|moldiest|moodiest|muddiest|muggiest|murkiest|mushiest|narrowest|nastiest|naughtiest|neatest|neediest|newest|nicest|niftiest|nimblest|noblest|noisiest|nosiest|numbest|nuttiest|obscurest|oddest|oiliest|oldest|orneriest|palest|paltriest|perkiest|pettiest|pinkest|plainest|pleasantest|pluckiest|plumpest|plushest|politest|poorest|portliest|prettiest|prickliest|primmest|prissiest|promptest|proudest|puffiest|puniest|purest|pushiest|quaintest|queasiest|queenliest|quickest|quietest|quirkiest|rainiest|rarest|rashest|raspiest|rattiest|rawest|reddest|remotest|richest|ripest|riskiest|ritziest|roomiest|rosiest|rottenest|roughest|roundest|rudest|rustiest|saddest|safest|saintliest|saltiest|sandiest|sanest|sappiest|sassiest|sauciest|scaliest|scantiest|scarcest|scariest|scraggliest|scrappiest|scratchiest|scrawniest|scruffiest|scummiest|securest|seediest|seemliest|serenest|severest|shabbiest|shadiest|shaggiest|shakiest|shallowest|sharpest|shiest|shiftiest|shiniest|shoddiest|shortest|showiest|shrewdest|shrillest|shyest|sickest|sickliest|silkiest|silliest|simplest|sincerest|sketchiest|skimpiest|skinniest|sleekest|sleepiest|slickest|sliest|slightest|slimiest|slimmest|slipperiest|sloppiest|slowest|smallest|smartest|smelliest|smoggiest|smokiest|smoothest|snappiest|sneakiest|snootiest|snottiest|snuggest|softest|soggiest|soonest|sorest|sorriest|sourest|sparsest|speediest|spiciest|spiffiest|spikiest|spookiest|spriest|spryest|squarest|squiggliest|stalest|starkest|stateliest|staunchest|steadiest|steepest|sternest|stickiest|stiffest|stillest|stingiest|stodgiest|stormiest|straggliest|straightest|strangest|strictest|strongest|stubbiest|stuffiest|sturdiest|subtlest|sulkiest|sunniest|surest|surliest|swankiest|sweatiest|sweetest|swiftest|tackiest|tallest|tamest|tangiest|tannest|tardiest|tartest|tastiest|tautest|teeniest|teensiest|teeny-tiniest|tersest|testiest|thickest|thinnest|thirstiest|thorniest|thriftiest|tidiest|tightest|timeliest|tiniest|toothiest|toughest|trashiest|trendiest|trickiest|trimmest|truest|trustiest|twitchiest|ugliest|unhappiest|unlikeliest|unluckiest|unruliest|vaguest|vainest|vilest|wackiest|wariest|warmest|wateriest|weakest|wealthiest|weariest|weediest|weirdest|wettest|whitest|wickedest|widest|wiggliest|wildest|windiest|wisest|wispiest|wittiest|wobbliest|wooziest|wordiest|worldliest|worthiest|wriest|wryest|yummiest|zaniest|zestiest|ablest|biggest|bravest|cleverest|fattest|greatest|hottest|kindest|noblest|saddest|smallest|sweetest|whitest|wisest|youngest)\b'), 'ADJ'),
    (re.compile(r'\b(?:most beautiful|most boring|most colorful|most comfortable|most complete|most cruel|most delicious|most difficult|most evil|most expensive|most famous|most foolish|most friendly|most generous|most important|most interesting|most modern|most nervous|most popular|most renowned|most tangled|most tilted|most tired|least energetic)\b'), 'ADJ')
]

common_superlatives = {
    'best', 'worst', 'furthest', 'farthest', 'least', 'most', 'latest', 'last', 'nearest', 'dearest'
}

# Get all sentences from the Brown corpus
sentences = brown.sents()

# Function to initialize Stanza pipeline
def initialize_pipeline():
    return stanza.Pipeline('en', processors='tokenize,pos,lemma', use_gpu=False)

# Function to process a batch of sentences
def process_sentence_batch(sentences_batch, nlp):
    batch_adjectives = set()
    batch_adverbs = set()
    for sentence in sentences_batch:
        sentence_text = ' '.join(sentence)
        sentence_text_no_punct = re.sub(r'[.,!?;:(){}\[\]\'"@#$%^&*+=|\\/<>\~`]', '', sentence_text)
        found_adjectives = set()
        found_adverbs = set()

        # Use regex patterns to find superlatives
        for pattern, pos in superlative_patterns:
            matches = pattern.findall(sentence_text_no_punct)
            if pos == 'ADJ':
                found_adjectives.update(match.lower() for match in matches)
            elif pos == 'ADV':
                found_adverbs.update(match.lower() for match in matches)

        # Match common superlatives
        found_adjectives.update(word for word in common_superlatives if word in sentence_text_no_punct.lower())

        # If no matches found, use Stanza to find superlatives
        if not found_adjectives and not found_adverbs:
            doc = nlp(sentence_text)
            for sent in doc.sentences:
                for word in sent.words:
                    if word.upos == 'ADJ' and word.feats and 'Degree=Sup' in word.feats:
                        found_adjectives.add(word.text.lower().replace(' ', '-'))
                    elif word.upos == 'ADV' and word.feats and 'Degree=Sup' in word.feats:
                        found_adverbs.add(word.text.lower().replace(' ', '-'))

        batch_adjectives.update(found_adjectives)
        batch_adverbs.update(found_adverbs)

    return batch_adjectives, batch_adverbs

# Function to process all sentences
def process_sentences(sentences, batch_size):
    adjectives = set()
    adverbs = set()

    nlp = initialize_pipeline()

    num_batches = len(sentences) // batch_size + (1 if len(sentences) % batch_size != 0 else 0)
    for i in range(num_batches):
        batch = sentences[i*batch_size:(i+1)*batch_size]
        batch_adjectives, batch_adverbs = process_sentence_batch(batch, nlp)
        adjectives.update(batch_adjectives)
        adverbs.update(batch_adverbs)

    return adjectives, adverbs

# Record start time
start_time = time.time()

# Process sentences
batch_size = 4096
superlative_adjectives, superlative_adverbs = process_sentences(sentences, batch_size)

# Record end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Print execution time
print("\nExecution Time:", execution_time, "seconds")

df_brown_superlative_adjectives = pd.DataFrame(list(superlative_adjectives), columns=['Word'])
df_brown_superlative_adverbs = pd.DataFrame(list(superlative_adverbs), columns=['Word'])

print("Superlative Adjectives:")
print(df_brown_superlative_adjectives.head(20))

print("\nSuperlative Adverbs:")
print(df_brown_superlative_adverbs.head(20))

output_dir = '/content/drive/My Drive/'
df_brown_superlative_adjectives = pd.DataFrame(list(superlative_adjectives), columns=['Word'])
df_brown_superlative_adverbs = pd.DataFrame(list(superlative_adverbs), columns=['Word'])

df_brown_superlative_adjectives.to_csv(os.path.join(output_dir, 'brown_superlative_adjectives.csv'), index=False)
df_brown_superlative_adverbs.to_csv(os.path.join(output_dir, 'brown_superlative_adverbs.csv'), index=False)

print("Superlative Adjectives saved to Google Drive.")
print("Superlative Adverbs saved to Google Drive.")

"""# webtext"""

nltk.download('punkt')

# Define superlative patterns
superlative_patterns = [
    (re.compile(r'\b\w+est\b'), 'ADJ'),
    (re.compile(r'\b\w+iest\b'), 'ADJ'),
    (re.compile(r'\b(?:most|least|best|worst)\s+\w+\b'), 'ADJ'),
    (re.compile(r'\b(?:most|least|best|worst)\s+\w+ly\b'), 'ADV'),
    (re.compile(r'\b(?:foremost|hindmost|inmost|innermost|nethermost|outmost|outermost|topmost|undermost|upmost|uppermost|utmost|uttermost)\b'), 'ADJ'),
    (re.compile(r'\b(?:ablest|angriest|baldest|battiest|beadiest|bitterest|blackest|blandest|blankest|bleakest|blondest|bloodiest|bluest|bluntest|blurriest|boldest|bossiest|bounciest|brainiest|brashest|brassiest|bravest|brawniest|breeziest|briefest|brightest|briskest|broadest|brownest|bubbliest|bulkiest|bumpiest|burliest|bushiest|busiest|calmest|chattiest|cheapest|cheekiest|cheeriest|chewiest|chilliest|choicest|choppiest|chubbiest|chunkiest|clammiest|classiest|cleanest|clearest|cleverest|closest|cloudiest|clumsiest|coarsest|coldest|coolest|corniest|coziest|crabbiest|craftiest|crankiest|craziest|creakiest|creamiest|creepiest|crispest|crispiest|crudest|cruelest|crumbliest|crunchiest|crustiest|cuddliest|curliest|curviest|cutest|daffiest|daintiest|dampest|dandiest|darkest|deadliest|deepest|densest|dingiest|dirtiest|dizziest|dreamiest|dreariest|dressiest|driest|droopiest|drowsiest|dullest|dumbest|dustiest|earliest|easiest|edgiest|eeriest|emptiest|evilest|faintest|fairest|falsest|fanciest|fattest|faultiest|feeblest|fewest|ficklest|fiercest|fieriest|filmiest|filthiest|finest|firmest|fittest|flabbiest|flakiest|flashiest|flattest|flimsiest|floppiest|floweriest|fluffiest|foamiest|foggiest|fondest|foolhardiest|frailest|frankest|freakiest|freest|freshest|friendliest|frilliest|friskiest|frostiest|frothiest|fruitiest|frumpiest|fullest|funniest|furriest|fussiest|fuzziest|gabbiest|gaudiest|gauntest|gawkiest|gentlest|ghastliest|giddiest|glassiest|gloomiest|glossiest|goofiest|grainiest|grandest|gravest|greasiest|greatest|greediest|greenest|grimiest|grittiest|groggiest|grossest|grouchiest|grubbiest|gruffest|grumpiest|guiltiest|gustiest|gutsiest|hairiest|handiest|handsomest|happiest|hardest|hardiest|harshest|hastiest|haughtiest|haziest|healthiest|heartiest|heaviest|heftiest|highest|hippest|hoarsest|hollowest|homeliest|hottest|hugest|humblest|hungriest|huskiest|iciest|ickiest|itchiest|itty-bittiest|jazziest|jerkiest|jolliest|juiciest|kindest|kindliest|kingliest|knobbiest|knottiest|laciest|largest|laziest|leanest|lengthiest|lightest|likeliest|littlest|liveliest|loneliest|longest|loosest|loudest|lousiest|loveliest|lowest|lowliest|luckiest|lumpiest|maddest|meanest|meekest|mellowest|merriest|messiest|mightiest|mildest|mistiest|moistest|moldiest|moodiest|muddiest|muggiest|murkiest|mushiest|narrowest|nastiest|naughtiest|neatest|neediest|newest|nicest|niftiest|nimblest|noblest|noisiest|nosiest|numbest|nuttiest|obscurest|oddest|oiliest|oldest|orneriest|palest|paltriest|perkiest|pettiest|pinkest|plainest|pleasantest|pluckiest|plumpest|plushest|politest|poorest|portliest|prettiest|prickliest|primmest|prissiest|promptest|proudest|puffiest|puniest|purest|pushiest|quaintest|queasiest|queenliest|quickest|quietest|quirkiest|rainiest|rarest|rashest|raspiest|rattiest|rawest|reddest|remotest|richest|ripest|riskiest|ritziest|roomiest|rosiest|rottenest|roughest|roundest|rudest|rustiest|saddest|safest|saintliest|saltiest|sandiest|sanest|sappiest|sassiest|sauciest|scaliest|scantiest|scarcest|scariest|scraggliest|scrappiest|scratchiest|scrawniest|scruffiest|scummiest|securest|seediest|seemliest|serenest|severest|shabbiest|shadiest|shaggiest|shakiest|shallowest|sharpest|shiest|shiftiest|shiniest|shoddiest|shortest|showiest|shrewdest|shrillest|shyest|sickest|sickliest|silkiest|silliest|simplest|sincerest|sketchiest|skimpiest|skinniest|sleekest|sleepiest|slickest|sliest|slightest|slimiest|slimmest|slipperiest|sloppiest|slowest|smallest|smartest|smelliest|smoggiest|smokiest|smoothest|snappiest|sneakiest|snootiest|snottiest|snuggest|softest|soggiest|soonest|sorest|sorriest|sourest|sparsest|speediest|spiciest|spiffiest|spikiest|spookiest|spriest|spryest|squarest|squiggliest|stalest|starkest|stateliest|staunchest|steadiest|steepest|sternest|stickiest|stiffest|stillest|stingiest|stodgiest|stormiest|straggliest|straightest|strangest|strictest|strongest|stubbiest|stuffiest|sturdiest|subtlest|sulkiest|sunniest|surest|surliest|swankiest|sweatiest|sweetest|swiftest|tackiest|tallest|tamest|tangiest|tannest|tardiest|tartest|tastiest|tautest|teeniest|teensiest|teeny-tiniest|tersest|testiest|thickest|thinnest|thirstiest|thorniest|thriftiest|tidiest|tightest|timeliest|tiniest|toothiest|toughest|trashiest|trendiest|trickiest|trimmest|truest|trustiest|twitchiest|ugliest|unhappiest|unlikeliest|unluckiest|unruliest|vaguest|vainest|vilest|wackiest|wariest|warmest|wateriest|weakest|wealthiest|weariest|weediest|weirdest|wettest|whitest|wickedest|widest|wiggliest|wildest|windiest|wisest|wispiest|wittiest|wobbliest|wooziest|wordiest|worldliest|worthiest|wriest|wryest|yummiest|zaniest|zestiest|ablest|biggest|bravest|cleverest|fattest|greatest|hottest|kindest|noblest|saddest|smallest|sweetest|whitest|wisest|youngest)\b'), 'ADJ'),
    (re.compile(r'\b(?:most beautiful|most boring|most colorful|most comfortable|most complete|most cruel|most delicious|most difficult|most evil|most expensive|most famous|most foolish|most friendly|most generous|most important|most interesting|most modern|most nervous|most popular|most renowned|most tangled|most tilted|most tired|least energetic)\b'), 'ADJ')
]

common_superlatives = {
    'best', 'worst', 'furthest', 'farthest', 'least', 'most', 'latest', 'last', 'nearest', 'dearest'
}

# Get all sentences from the Brown corpus
sentences = webtext.sents()

# Function to initialize Stanza pipeline
def initialize_pipeline():
    return stanza.Pipeline('en', processors='tokenize,pos,lemma', use_gpu=False)

# Function to process a batch of sentences
def process_sentence_batch(sentences_batch, nlp):
    batch_adjectives = set()
    batch_adverbs = set()
    for sentence in sentences_batch:
        sentence_text = ' '.join(sentence)
        sentence_text_no_punct = re.sub(r'[.,!?;:(){}\[\]\'"@#$%^&*+=|\\/<>\~`]', '', sentence_text)
        found_adjectives = set()
        found_adverbs = set()

        # Use regex patterns to find superlatives
        for pattern, pos in superlative_patterns:
            matches = pattern.findall(sentence_text_no_punct)
            if pos == 'ADJ':
                found_adjectives.update(match.lower() for match in matches)
            elif pos == 'ADV':
                found_adverbs.update(match.lower() for match in matches)

        # Match common superlatives
        found_adjectives.update(word for word in common_superlatives if word in sentence_text_no_punct.lower())

        # If no matches found, use Stanza to find superlatives
        if not found_adjectives and not found_adverbs:
            doc = nlp(sentence_text)
            for sent in doc.sentences:
                for word in sent.words:
                    if word.upos == 'ADJ' and word.feats and 'Degree=Sup' in word.feats:
                        found_adjectives.add(word.text.lower().replace(' ', '-'))
                    elif word.upos == 'ADV' and word.feats and 'Degree=Sup' in word.feats:
                        found_adverbs.add(word.text.lower().replace(' ', '-'))

        batch_adjectives.update(found_adjectives)
        batch_adverbs.update(found_adverbs)

    return batch_adjectives, batch_adverbs

# Function to process all sentences
def process_sentences(sentences, batch_size):
    adjectives = set()
    adverbs = set()

    nlp = initialize_pipeline()

    num_batches = len(sentences) // batch_size + (1 if len(sentences) % batch_size != 0 else 0)
    for i in range(num_batches):
        batch = sentences[i*batch_size:(i+1)*batch_size]
        batch_adjectives, batch_adverbs = process_sentence_batch(batch, nlp)
        adjectives.update(batch_adjectives)
        adverbs.update(batch_adverbs)

    return adjectives, adverbs

# Record start time
start_time = time.time()

# Process sentences
batch_size = 4096
superlative_adjectives, superlative_adverbs = process_sentences(sentences, batch_size)

# Record end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Print execution time
print("\nExecution Time:", execution_time, "seconds")

df_webtext0_superlative_adjectives = pd.DataFrame(list(superlative_adjectives), columns=['Word'])
df_webtext0_superlative_adverbs = pd.DataFrame(list(superlative_adverbs), columns=['Word'])

print("Superlative Adjectives:")
print(df_webtext0_superlative_adjectives.head(20))

print("\nSuperlative Adverbs:")
print(df_webtext0_superlative_adverbs.head(20))

# save to csv
output_dir = '/content/drive/My Drive/'
df_webtext0_superlative_adjectives = pd.DataFrame(list(superlative_adjectives), columns=['Word'])
df_webtext0_superlative_adverbs = pd.DataFrame(list(superlative_adverbs), columns=['Word'])

df_webtext0_superlative_adjectives.to_csv(os.path.join(output_dir, 'webtext0_superlative_adjectives.csv'), index=False)
df_webtext0_superlative_adverbs.to_csv(os.path.join(output_dir, 'webtext0_superlative_adverbs.csv'), index=False)

print("Superlative Adjectives saved to Google Drive.")
print("Superlative Adverbs saved to Google Drive.")

"""# Reuters"""

# Define superlative patterns
superlative_patterns = [
    (re.compile(r'\b\w+est\b'), 'ADJ'),
    (re.compile(r'\b\w+iest\b'), 'ADJ'),
    (re.compile(r'\b(?:most|least|best|worst)\s+\w+\b'), 'ADJ'),
    (re.compile(r'\b(?:most|least|best|worst)\s+\w+ly\b'), 'ADV'),
    (re.compile(r'\b(?:foremost|hindmost|inmost|innermost|nethermost|outmost|outermost|topmost|undermost|upmost|uppermost|utmost|uttermost)\b'), 'ADJ'),
    (re.compile(r'\b(?:ablest|angriest|baldest|battiest|beadiest|bitterest|blackest|blandest|blankest|bleakest|blondest|bloodiest|bluest|bluntest|blurriest|boldest|bossiest|bounciest|brainiest|brashest|brassiest|bravest|brawniest|breeziest|briefest|brightest|briskest|broadest|brownest|bubbliest|bulkiest|bumpiest|burliest|bushiest|busiest|calmest|chattiest|cheapest|cheekiest|cheeriest|chewiest|chilliest|choicest|choppiest|chubbiest|chunkiest|clammiest|classiest|cleanest|clearest|cleverest|closest|cloudiest|clumsiest|coarsest|coldest|coolest|corniest|coziest|crabbiest|craftiest|crankiest|craziest|creakiest|creamiest|creepiest|crispest|crispiest|crudest|cruelest|crumbliest|crunchiest|crustiest|cuddliest|curliest|curviest|cutest|daffiest|daintiest|dampest|dandiest|darkest|deadliest|deepest|densest|dingiest|dirtiest|dizziest|dreamiest|dreariest|dressiest|driest|droopiest|drowsiest|dullest|dumbest|dustiest|earliest|easiest|edgiest|eeriest|emptiest|evilest|faintest|fairest|falsest|fanciest|fattest|faultiest|feeblest|fewest|ficklest|fiercest|fieriest|filmiest|filthiest|finest|firmest|fittest|flabbiest|flakiest|flashiest|flattest|flimsiest|floppiest|floweriest|fluffiest|foamiest|foggiest|fondest|foolhardiest|frailest|frankest|freakiest|freest|freshest|friendliest|frilliest|friskiest|frostiest|frothiest|fruitiest|frumpiest|fullest|funniest|furriest|fussiest|fuzziest|gabbiest|gaudiest|gauntest|gawkiest|gentlest|ghastliest|giddiest|glassiest|gloomiest|glossiest|goofiest|grainiest|grandest|gravest|greasiest|greatest|greediest|greenest|grimiest|grittiest|groggiest|grossest|grouchiest|grubbiest|gruffest|grumpiest|guiltiest|gustiest|gutsiest|hairiest|handiest|handsomest|happiest|hardest|hardiest|harshest|hastiest|haughtiest|haziest|healthiest|heartiest|heaviest|heftiest|highest|hippest|hoarsest|hollowest|homeliest|hottest|hugest|humblest|hungriest|huskiest|iciest|ickiest|itchiest|itty-bittiest|jazziest|jerkiest|jolliest|juiciest|kindest|kindliest|kingliest|knobbiest|knottiest|laciest|largest|laziest|leanest|lengthiest|lightest|likeliest|littlest|liveliest|loneliest|longest|loosest|loudest|lousiest|loveliest|lowest|lowliest|luckiest|lumpiest|maddest|meanest|meekest|mellowest|merriest|messiest|mightiest|mildest|mistiest|moistest|moldiest|moodiest|muddiest|muggiest|murkiest|mushiest|narrowest|nastiest|naughtiest|neatest|neediest|newest|nicest|niftiest|nimblest|noblest|noisiest|nosiest|numbest|nuttiest|obscurest|oddest|oiliest|oldest|orneriest|palest|paltriest|perkiest|pettiest|pinkest|plainest|pleasantest|pluckiest|plumpest|plushest|politest|poorest|portliest|prettiest|prickliest|primmest|prissiest|promptest|proudest|puffiest|puniest|purest|pushiest|quaintest|queasiest|queenliest|quickest|quietest|quirkiest|rainiest|rarest|rashest|raspiest|rattiest|rawest|reddest|remotest|richest|ripest|riskiest|ritziest|roomiest|rosiest|rottenest|roughest|roundest|rudest|rustiest|saddest|safest|saintliest|saltiest|sandiest|sanest|sappiest|sassiest|sauciest|scaliest|scantiest|scarcest|scariest|scraggliest|scrappiest|scratchiest|scrawniest|scruffiest|scummiest|securest|seediest|seemliest|serenest|severest|shabbiest|shadiest|shaggiest|shakiest|shallowest|sharpest|shiest|shiftiest|shiniest|shoddiest|shortest|showiest|shrewdest|shrillest|shyest|sickest|sickliest|silkiest|silliest|simplest|sincerest|sketchiest|skimpiest|skinniest|sleekest|sleepiest|slickest|sliest|slightest|slimiest|slimmest|slipperiest|sloppiest|slowest|smallest|smartest|smelliest|smoggiest|smokiest|smoothest|snappiest|sneakiest|snootiest|snottiest|snuggest|softest|soggiest|soonest|sorest|sorriest|sourest|sparsest|speediest|spiciest|spiffiest|spikiest|spookiest|spriest|spryest|squarest|squiggliest|stalest|starkest|stateliest|staunchest|steadiest|steepest|sternest|stickiest|stiffest|stillest|stingiest|stodgiest|stormiest|straggliest|straightest|strangest|strictest|strongest|stubbiest|stuffiest|sturdiest|subtlest|sulkiest|sunniest|surest|surliest|swankiest|sweatiest|sweetest|swiftest|tackiest|tallest|tamest|tangiest|tannest|tardiest|tartest|tastiest|tautest|teeniest|teensiest|teeny-tiniest|tersest|testiest|thickest|thinnest|thirstiest|thorniest|thriftiest|tidiest|tightest|timeliest|tiniest|toothiest|toughest|trashiest|trendiest|trickiest|trimmest|truest|trustiest|twitchiest|ugliest|unhappiest|unlikeliest|unluckiest|unruliest|vaguest|vainest|vilest|wackiest|wariest|warmest|wateriest|weakest|wealthiest|weariest|weediest|weirdest|wettest|whitest|wickedest|widest|wiggliest|wildest|windiest|wisest|wispiest|wittiest|wobbliest|wooziest|wordiest|worldliest|worthiest|wriest|wryest|yummiest|zaniest|zestiest|ablest|biggest|bravest|cleverest|fattest|greatest|hottest|kindest|noblest|saddest|smallest|sweetest|whitest|wisest|youngest)\b'), 'ADJ'),
    (re.compile(r'\b(?:most beautiful|most boring|most colorful|most comfortable|most complete|most cruel|most delicious|most difficult|most evil|most expensive|most famous|most foolish|most friendly|most generous|most important|most interesting|most modern|most nervous|most popular|most renowned|most tangled|most tilted|most tired|least energetic)\b'), 'ADJ')
]

common_superlatives = {
    'best', 'worst', 'furthest', 'farthest', 'least', 'most', 'latest', 'last', 'nearest', 'dearest'
}

# Get all sentences from the Reuters corpus
sentences = reuters.sents()

# Function to initialize Stanza pipeline
def initialize_pipeline():
    return stanza.Pipeline('en', processors='tokenize,pos,lemma', use_gpu=False)

# Function to process a batch of sentences
def process_sentence_batch(sentences_batch, nlp):
    batch_adjectives = set()
    batch_adverbs = set()
    for sentence in sentences_batch:
        sentence_text = ' '.join(sentence)
        sentence_text_no_punct = re.sub(r'[.,!?;:(){}\[\]\'"@#$%^&*+=|\\/<>\~`]', '', sentence_text)
        found_adjectives = set()
        found_adverbs = set()

        # Use regex patterns to find superlatives
        for pattern, pos in superlative_patterns:
            matches = pattern.findall(sentence_text_no_punct)
            if pos == 'ADJ':
                found_adjectives.update(match.lower() for match in matches)
            elif pos == 'ADV':
                found_adverbs.update(match.lower() for match in matches)

        # Match common superlatives
        found_adjectives.update(word for word in common_superlatives if word in sentence_text_no_punct.lower())

        # If no matches found, use Stanza to find superlatives
        if not found_adjectives and not found_adverbs:
            doc = nlp(sentence_text)
            for sent in doc.sentences:
                for word in sent.words:
                    if word.upos == 'ADJ' and word.feats and 'Degree=Sup' in word.feats:
                        found_adjectives.add(word.text.lower().replace(' ', '-'))
                    elif word.upos == 'ADV' and word.feats and 'Degree=Sup' in word.feats:
                        found_adverbs.add(word.text.lower().replace(' ', '-'))

        batch_adjectives.update(found_adjectives)
        batch_adverbs.update(found_adverbs)

    return batch_adjectives, batch_adverbs

# Function to process all sentences
def process_sentences(sentences, batch_size):
    adjectives = set()
    adverbs = set()

    nlp = initialize_pipeline()

    num_batches = len(sentences) // batch_size + (1 if len(sentences) % batch_size != 0 else 0)
    for i in range(num_batches):
        batch = sentences[i*batch_size:(i+1)*batch_size]
        batch_adjectives, batch_adverbs = process_sentence_batch(batch, nlp)
        adjectives.update(batch_adjectives)
        adverbs.update(batch_adverbs)

    return adjectives, adverbs

# Record start time
start_time = time.time()

# Process sentences
batch_size = 4096
superlative_adjectives, superlative_adverbs = process_sentences(sentences, batch_size)

# Record end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Print execution time
print("\nExecution Time:", execution_time, "seconds")

df_webtext_superlative_adjectives = pd.DataFrame(list(superlative_adjectives), columns=['Word'])
df_webtext_superlative_adverbs = pd.DataFrame(list(superlative_adverbs), columns=['Word'])

print("Superlative Adjectives:")
print(df_webtext_superlative_adjectives.head(20))

print("\nSuperlative Adverbs:")
print(df_webtext_superlative_adverbs.head(20))

output_dir = '/content/drive/My Drive/'
df_reuters_superlative_adjectives = pd.DataFrame(list(superlative_adjectives), columns=['Word'])
df_reuters_superlative_adverbs = pd.DataFrame(list(superlative_adverbs), columns=['Word'])

df_reuters_superlative_adjectives.to_csv(os.path.join(output_dir, 'reuters_superlative_adjectives.csv'), index=False)
df_reuters_superlative_adverbs.to_csv(os.path.join(output_dir, 'reuters_superlative_adverbs.csv'), index=False)

print("Superlative Adjectives saved to Google Drive.")
print("Superlative Adverbs saved to Google Drive.")

"""# Gutenberg"""

# Define superlative patterns
superlative_patterns = [
    (re.compile(r'\b\w+est\b'), 'ADJ'),
    (re.compile(r'\b\w+iest\b'), 'ADJ'),
    (re.compile(r'\b(?:most|least|best|worst)\s+\w+\b'), 'ADJ'),
    (re.compile(r'\b(?:most|least|best|worst)\s+\w+ly\b'), 'ADV'),
    (re.compile(r'\b(?:foremost|hindmost|inmost|innermost|nethermost|outmost|outermost|topmost|undermost|upmost|uppermost|utmost|uttermost)\b'), 'ADJ'),
    (re.compile(r'\b(?:ablest|angriest|baldest|battiest|beadiest|bitterest|blackest|blandest|blankest|bleakest|blondest|bloodiest|bluest|bluntest|blurriest|boldest|bossiest|bounciest|brainiest|brashest|brassiest|bravest|brawniest|breeziest|briefest|brightest|briskest|broadest|brownest|bubbliest|bulkiest|bumpiest|burliest|bushiest|busiest|calmest|chattiest|cheapest|cheekiest|cheeriest|chewiest|chilliest|choicest|choppiest|chubbiest|chunkiest|clammiest|classiest|cleanest|clearest|cleverest|closest|cloudiest|clumsiest|coarsest|coldest|coolest|corniest|coziest|crabbiest|craftiest|crankiest|craziest|creakiest|creamiest|creepiest|crispest|crispiest|crudest|cruelest|crumbliest|crunchiest|crustiest|cuddliest|curliest|curviest|cutest|daffiest|daintiest|dampest|dandiest|darkest|deadliest|deepest|densest|dingiest|dirtiest|dizziest|dreamiest|dreariest|dressiest|driest|droopiest|drowsiest|dullest|dumbest|dustiest|earliest|easiest|edgiest|eeriest|emptiest|evilest|faintest|fairest|falsest|fanciest|fattest|faultiest|feeblest|fewest|ficklest|fiercest|fieriest|filmiest|filthiest|finest|firmest|fittest|flabbiest|flakiest|flashiest|flattest|flimsiest|floppiest|floweriest|fluffiest|foamiest|foggiest|fondest|foolhardiest|frailest|frankest|freakiest|freest|freshest|friendliest|frilliest|friskiest|frostiest|frothiest|fruitiest|frumpiest|fullest|funniest|furriest|fussiest|fuzziest|gabbiest|gaudiest|gauntest|gawkiest|gentlest|ghastliest|giddiest|glassiest|gloomiest|glossiest|goofiest|grainiest|grandest|gravest|greasiest|greatest|greediest|greenest|grimiest|grittiest|groggiest|grossest|grouchiest|grubbiest|gruffest|grumpiest|guiltiest|gustiest|gutsiest|hairiest|handiest|handsomest|happiest|hardest|hardiest|harshest|hastiest|haughtiest|haziest|healthiest|heartiest|heaviest|heftiest|highest|hippest|hoarsest|hollowest|homeliest|hottest|hugest|humblest|hungriest|huskiest|iciest|ickiest|itchiest|itty-bittiest|jazziest|jerkiest|jolliest|juiciest|kindest|kindliest|kingliest|knobbiest|knottiest|laciest|largest|laziest|leanest|lengthiest|lightest|likeliest|littlest|liveliest|loneliest|longest|loosest|loudest|lousiest|loveliest|lowest|lowliest|luckiest|lumpiest|maddest|meanest|meekest|mellowest|merriest|messiest|mightiest|mildest|mistiest|moistest|moldiest|moodiest|muddiest|muggiest|murkiest|mushiest|narrowest|nastiest|naughtiest|neatest|neediest|newest|nicest|niftiest|nimblest|noblest|noisiest|nosiest|numbest|nuttiest|obscurest|oddest|oiliest|oldest|orneriest|palest|paltriest|perkiest|pettiest|pinkest|plainest|pleasantest|pluckiest|plumpest|plushest|politest|poorest|portliest|prettiest|prickliest|primmest|prissiest|promptest|proudest|puffiest|puniest|purest|pushiest|quaintest|queasiest|queenliest|quickest|quietest|quirkiest|rainiest|rarest|rashest|raspiest|rattiest|rawest|reddest|remotest|richest|ripest|riskiest|ritziest|roomiest|rosiest|rottenest|roughest|roundest|rudest|rustiest|saddest|safest|saintliest|saltiest|sandiest|sanest|sappiest|sassiest|sauciest|scaliest|scantiest|scarcest|scariest|scraggliest|scrappiest|scratchiest|scrawniest|scruffiest|scummiest|securest|seediest|seemliest|serenest|severest|shabbiest|shadiest|shaggiest|shakiest|shallowest|sharpest|shiest|shiftiest|shiniest|shoddiest|shortest|showiest|shrewdest|shrillest|shyest|sickest|sickliest|silkiest|silliest|simplest|sincerest|sketchiest|skimpiest|skinniest|sleekest|sleepiest|slickest|sliest|slightest|slimiest|slimmest|slipperiest|sloppiest|slowest|smallest|smartest|smelliest|smoggiest|smokiest|smoothest|snappiest|sneakiest|snootiest|snottiest|snuggest|softest|soggiest|soonest|sorest|sorriest|sourest|sparsest|speediest|spiciest|spiffiest|spikiest|spookiest|spriest|spryest|squarest|squiggliest|stalest|starkest|stateliest|staunchest|steadiest|steepest|sternest|stickiest|stiffest|stillest|stingiest|stodgiest|stormiest|straggliest|straightest|strangest|strictest|strongest|stubbiest|stuffiest|sturdiest|subtlest|sulkiest|sunniest|surest|surliest|swankiest|sweatiest|sweetest|swiftest|tackiest|tallest|tamest|tangiest|tannest|tardiest|tartest|tastiest|tautest|teeniest|teensiest|teeny-tiniest|tersest|testiest|thickest|thinnest|thirstiest|thorniest|thriftiest|tidiest|tightest|timeliest|tiniest|toothiest|toughest|trashiest|trendiest|trickiest|trimmest|truest|trustiest|twitchiest|ugliest|unhappiest|unlikeliest|unluckiest|unruliest|vaguest|vainest|vilest|wackiest|wariest|warmest|wateriest|weakest|wealthiest|weariest|weediest|weirdest|wettest|whitest|wickedest|widest|wiggliest|wildest|windiest|wisest|wispiest|wittiest|wobbliest|wooziest|wordiest|worldliest|worthiest|wriest|wryest|yummiest|zaniest|zestiest|ablest|biggest|bravest|cleverest|fattest|greatest|hottest|kindest|noblest|saddest|smallest|sweetest|whitest|wisest|youngest)\b'), 'ADJ'),
    (re.compile(r'\b(?:most beautiful|most boring|most colorful|most comfortable|most complete|most cruel|most delicious|most difficult|most evil|most expensive|most famous|most foolish|most friendly|most generous|most important|most interesting|most modern|most nervous|most popular|most renowned|most tangled|most tilted|most tired|least energetic)\b'), 'ADJ')
]

common_superlatives = {
    'best', 'worst', 'furthest', 'farthest', 'least', 'most', 'latest', 'last', 'nearest', 'dearest'
}

# Get all sentences from the Gutenberg corpus
sentences = reuters.sents()

# Function to initialize Stanza pipeline
def initialize_pipeline():
    return stanza.Pipeline('en', processors='tokenize,pos,lemma', use_gpu=False)

# Function to process a batch of sentences
def process_sentence_batch(sentences_batch, nlp):
    batch_adjectives = set()
    batch_adverbs = set()
    for sentence in sentences_batch:
        sentence_text = ' '.join(sentence)
        sentence_text_no_punct = re.sub(r'[.,!?;:(){}\[\]\'"@#$%^&*+=|\\/<>\~`]', '', sentence_text)
        found_adjectives = set()
        found_adverbs = set()

        # Use regex patterns to find superlatives
        for pattern, pos in superlative_patterns:
            matches = pattern.findall(sentence_text_no_punct)
            if pos == 'ADJ':
                found_adjectives.update(match.lower() for match in matches)
            elif pos == 'ADV':
                found_adverbs.update(match.lower() for match in matches)

        # Match common superlatives
        found_adjectives.update(word for word in common_superlatives if word in sentence_text_no_punct.lower())

        # If no matches found, use Stanza to find superlatives
        if not found_adjectives and not found_adverbs:
            doc = nlp(sentence_text)
            for sent in doc.sentences:
                for word in sent.words:
                    if word.upos == 'ADJ' and word.feats and 'Degree=Sup' in word.feats:
                        found_adjectives.add(word.text.lower().replace(' ', '-'))
                    elif word.upos == 'ADV' and word.feats and 'Degree=Sup' in word.feats:
                        found_adverbs.add(word.text.lower().replace(' ', '-'))

        batch_adjectives.update(found_adjectives)
        batch_adverbs.update(found_adverbs)

    return batch_adjectives, batch_adverbs

# Function to process all sentences
def process_sentences(sentences, batch_size):
    adjectives = set()
    adverbs = set()

    nlp = initialize_pipeline()

    num_batches = len(sentences) // batch_size + (1 if len(sentences) % batch_size != 0 else 0)
    for i in range(num_batches):
        batch = sentences[i*batch_size:(i+1)*batch_size]
        batch_adjectives, batch_adverbs = process_sentence_batch(batch, nlp)
        adjectives.update(batch_adjectives)
        adverbs.update(batch_adverbs)

    return adjectives, adverbs

# Record start time
start_time = time.time()

# Process sentences
batch_size = 4096
superlative_adjectives, superlative_adverbs = process_sentences(sentences, batch_size)

# Record end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Print execution time
print("\nExecution Time:", execution_time, "seconds")

df_webtext_superlative_adjectives = pd.DataFrame(list(superlative_adjectives), columns=['Word'])
df_webtext_superlative_adverbs = pd.DataFrame(list(superlative_adverbs), columns=['Word'])

print("Superlative Adjectives:")
print(df_webtext_superlative_adjectives.head(20))

print("\nSuperlative Adverbs:")
print(df_webtext_superlative_adverbs.head(20))

output_dir = '/content/drive/My Drive/'
df_gutenberg_superlative_adjectives = pd.DataFrame(list(superlative_adjectives), columns=['Word'])
df_gutenberg_superlative_adverbs = pd.DataFrame(list(superlative_adverbs), columns=['Word'])

df_gutenberg_superlative_adjectives.to_csv(os.path.join(output_dir, 'gutenberg_superlative_adjectives.csv'), index=False)
df_gutenberg_superlative_adverbs.to_csv(os.path.join(output_dir, 'gutenberg_superlative_adverbs.csv'), index=False)

print("Superlative Adjectives saved to Google Drive.")
print("Superlative Adverbs saved to Google Drive.")

"""# Movie_Reviews"""

# Define superlative patterns
superlative_patterns = [
    (re.compile(r'\b\w+est\b'), 'ADJ'),
    (re.compile(r'\b\w+iest\b'), 'ADJ'),
    (re.compile(r'\b(?:most|least|best|worst)\s+\w+\b'), 'ADJ'),
    (re.compile(r'\b(?:most|least|best|worst)\s+\w+ly\b'), 'ADV'),
    (re.compile(r'\b(?:foremost|hindmost|inmost|innermost|nethermost|outmost|outermost|topmost|undermost|upmost|uppermost|utmost|uttermost)\b'), 'ADJ'),
    (re.compile(r'\b(?:ablest|angriest|baldest|battiest|beadiest|bitterest|blackest|blandest|blankest|bleakest|blondest|bloodiest|bluest|bluntest|blurriest|boldest|bossiest|bounciest|brainiest|brashest|brassiest|bravest|brawniest|breeziest|briefest|brightest|briskest|broadest|brownest|bubbliest|bulkiest|bumpiest|burliest|bushiest|busiest|calmest|chattiest|cheapest|cheekiest|cheeriest|chewiest|chilliest|choicest|choppiest|chubbiest|chunkiest|clammiest|classiest|cleanest|clearest|cleverest|closest|cloudiest|clumsiest|coarsest|coldest|coolest|corniest|coziest|crabbiest|craftiest|crankiest|craziest|creakiest|creamiest|creepiest|crispest|crispiest|crudest|cruelest|crumbliest|crunchiest|crustiest|cuddliest|curliest|curviest|cutest|daffiest|daintiest|dampest|dandiest|darkest|deadliest|deepest|densest|dingiest|dirtiest|dizziest|dreamiest|dreariest|dressiest|driest|droopiest|drowsiest|dullest|dumbest|dustiest|earliest|easiest|edgiest|eeriest|emptiest|evilest|faintest|fairest|falsest|fanciest|fattest|faultiest|feeblest|fewest|ficklest|fiercest|fieriest|filmiest|filthiest|finest|firmest|fittest|flabbiest|flakiest|flashiest|flattest|flimsiest|floppiest|floweriest|fluffiest|foamiest|foggiest|fondest|foolhardiest|frailest|frankest|freakiest|freest|freshest|friendliest|frilliest|friskiest|frostiest|frothiest|fruitiest|frumpiest|fullest|funniest|furriest|fussiest|fuzziest|gabbiest|gaudiest|gauntest|gawkiest|gentlest|ghastliest|giddiest|glassiest|gloomiest|glossiest|goofiest|grainiest|grandest|gravest|greasiest|greatest|greediest|greenest|grimiest|grittiest|groggiest|grossest|grouchiest|grubbiest|gruffest|grumpiest|guiltiest|gustiest|gutsiest|hairiest|handiest|handsomest|happiest|hardest|hardiest|harshest|hastiest|haughtiest|haziest|healthiest|heartiest|heaviest|heftiest|highest|hippest|hoarsest|hollowest|homeliest|hottest|hugest|humblest|hungriest|huskiest|iciest|ickiest|itchiest|itty-bittiest|jazziest|jerkiest|jolliest|juiciest|kindest|kindliest|kingliest|knobbiest|knottiest|laciest|largest|laziest|leanest|lengthiest|lightest|likeliest|littlest|liveliest|loneliest|longest|loosest|loudest|lousiest|loveliest|lowest|lowliest|luckiest|lumpiest|maddest|meanest|meekest|mellowest|merriest|messiest|mightiest|mildest|mistiest|moistest|moldiest|moodiest|muddiest|muggiest|murkiest|mushiest|narrowest|nastiest|naughtiest|neatest|neediest|newest|nicest|niftiest|nimblest|noblest|noisiest|nosiest|numbest|nuttiest|obscurest|oddest|oiliest|oldest|orneriest|palest|paltriest|perkiest|pettiest|pinkest|plainest|pleasantest|pluckiest|plumpest|plushest|politest|poorest|portliest|prettiest|prickliest|primmest|prissiest|promptest|proudest|puffiest|puniest|purest|pushiest|quaintest|queasiest|queenliest|quickest|quietest|quirkiest|rainiest|rarest|rashest|raspiest|rattiest|rawest|reddest|remotest|richest|ripest|riskiest|ritziest|roomiest|rosiest|rottenest|roughest|roundest|rudest|rustiest|saddest|safest|saintliest|saltiest|sandiest|sanest|sappiest|sassiest|sauciest|scaliest|scantiest|scarcest|scariest|scraggliest|scrappiest|scratchiest|scrawniest|scruffiest|scummiest|securest|seediest|seemliest|serenest|severest|shabbiest|shadiest|shaggiest|shakiest|shallowest|sharpest|shiest|shiftiest|shiniest|shoddiest|shortest|showiest|shrewdest|shrillest|shyest|sickest|sickliest|silkiest|silliest|simplest|sincerest|sketchiest|skimpiest|skinniest|sleekest|sleepiest|slickest|sliest|slightest|slimiest|slimmest|slipperiest|sloppiest|slowest|smallest|smartest|smelliest|smoggiest|smokiest|smoothest|snappiest|sneakiest|snootiest|snottiest|snuggest|softest|soggiest|soonest|sorest|sorriest|sourest|sparsest|speediest|spiciest|spiffiest|spikiest|spookiest|spriest|spryest|squarest|squiggliest|stalest|starkest|stateliest|staunchest|steadiest|steepest|sternest|stickiest|stiffest|stillest|stingiest|stodgiest|stormiest|straggliest|straightest|strangest|strictest|strongest|stubbiest|stuffiest|sturdiest|subtlest|sulkiest|sunniest|surest|surliest|swankiest|sweatiest|sweetest|swiftest|tackiest|tallest|tamest|tangiest|tannest|tardiest|tartest|tastiest|tautest|teeniest|teensiest|teeny-tiniest|tersest|testiest|thickest|thinnest|thirstiest|thorniest|thriftiest|tidiest|tightest|timeliest|tiniest|toothiest|toughest|trashiest|trendiest|trickiest|trimmest|truest|trustiest|twitchiest|ugliest|unhappiest|unlikeliest|unluckiest|unruliest|vaguest|vainest|vilest|wackiest|wariest|warmest|wateriest|weakest|wealthiest|weariest|weediest|weirdest|wettest|whitest|wickedest|widest|wiggliest|wildest|windiest|wisest|wispiest|wittiest|wobbliest|wooziest|wordiest|worldliest|worthiest|wriest|wryest|yummiest|zaniest|zestiest|ablest|biggest|bravest|cleverest|fattest|greatest|hottest|kindest|noblest|saddest|smallest|sweetest|whitest|wisest|youngest)\b'), 'ADJ'),
    (re.compile(r'\b(?:most beautiful|most boring|most colorful|most comfortable|most complete|most cruel|most delicious|most difficult|most evil|most expensive|most famous|most foolish|most friendly|most generous|most important|most interesting|most modern|most nervous|most popular|most renowned|most tangled|most tilted|most tired|least energetic)\b'), 'ADJ')
]

common_superlatives = {
    'best', 'worst', 'furthest', 'farthest', 'least', 'most', 'latest', 'last', 'nearest', 'dearest'
}

# Get all sentences from the movie_reviews corpus
sentences = reuters.sents()

# Function to initialize Stanza pipeline
def initialize_pipeline():
    return stanza.Pipeline('en', processors='tokenize,pos,lemma', use_gpu=False)

# Function to process a batch of sentences
def process_sentence_batch(sentences_batch, nlp):
    batch_adjectives = set()
    batch_adverbs = set()
    for sentence in sentences_batch:
        sentence_text = ' '.join(sentence)
        sentence_text_no_punct = re.sub(r'[.,!?;:(){}\[\]\'"@#$%^&*+=|\\/<>\~`]', '', sentence_text)
        found_adjectives = set()
        found_adverbs = set()

        # Use regex patterns to find superlatives
        for pattern, pos in superlative_patterns:
            matches = pattern.findall(sentence_text_no_punct)
            if pos == 'ADJ':
                found_adjectives.update(match.lower() for match in matches)
            elif pos == 'ADV':
                found_adverbs.update(match.lower() for match in matches)

        # Match common superlatives
        found_adjectives.update(word for word in common_superlatives if word in sentence_text_no_punct.lower())

        # If no matches found, use Stanza to find superlatives
        if not found_adjectives and not found_adverbs:
            doc = nlp(sentence_text)
            for sent in doc.sentences:
                for word in sent.words:
                    if word.upos == 'ADJ' and word.feats and 'Degree=Sup' in word.feats:
                        found_adjectives.add(word.text.lower().replace(' ', '-'))
                    elif word.upos == 'ADV' and word.feats and 'Degree=Sup' in word.feats:
                        found_adverbs.add(word.text.lower().replace(' ', '-'))

        batch_adjectives.update(found_adjectives)
        batch_adverbs.update(found_adverbs)

    return batch_adjectives, batch_adverbs

# Function to process all sentences
def process_sentences(sentences, batch_size):
    adjectives = set()
    adverbs = set()

    nlp = initialize_pipeline()

    num_batches = len(sentences) // batch_size + (1 if len(sentences) % batch_size != 0 else 0)
    for i in range(num_batches):
        batch = sentences[i*batch_size:(i+1)*batch_size]
        batch_adjectives, batch_adverbs = process_sentence_batch(batch, nlp)
        adjectives.update(batch_adjectives)
        adverbs.update(batch_adverbs)

    return adjectives, adverbs

# Record start time
start_time = time.time()

# Process sentences
batch_size = 4096
superlative_adjectives, superlative_adverbs = process_sentences(sentences, batch_size)

# Record end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Print execution time
print("\nExecution Time:", execution_time, "seconds")

df_movie_reviews_superlative_adjectives = pd.DataFrame(list(superlative_adjectives), columns=['Word'])
df_movie_reviews_superlative_adverbs = pd.DataFrame(list(superlative_adverbs), columns=['Word'])

print("Superlative Adjectives:")
print(df_movie_reviews_superlative_adjectives.head(20))

print("\nSuperlative Adverbs:")
print(df_movie_reviews_superlative_adverbs.head(20))

output_dir = '/content/drive/My Drive/'
df_movie_reviews_superlative_adjectives = pd.DataFrame(list(superlative_adjectives), columns=['Word'])
df_movie_reviews_superlative_adverbs = pd.DataFrame(list(superlative_adverbs), columns=['Word'])

df_movie_reviews_superlative_adjectives.to_csv(os.path.join(output_dir, 'movie_reviews_superlative_adjectives.csv'), index=False)
df_movie_reviews_superlative_adverbs.to_csv(os.path.join(output_dir, 'movie_reviews_superlative_adverbs.csv'), index=False)

print("Superlative Adjectives saved to Google Drive.")
print("Superlative Adverbs saved to Google Drive.")
