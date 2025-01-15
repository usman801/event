import csv
import time
from newspaper import Article
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
import re
from collections import defaultdict

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')

# Define list of news article URLs
urls = [  
    'https://www.dawn.com/news/1878485/kp-to-launch-polio-drive-on-dec-16-amid-challenges',
'https://www.dawn.com/news/1878521/senate-unanimously-passes-national-forensic-agency-bill-2024',
'https://www.dawn.com/news/1878176/kp-govt-forms-working-group-for-climate-change-initiative',
'https://www.dawn.com/news/1878326/amid-ongoing-digital-law-reforms-govt-proposes-new-central-cybercrime-forensics-agency',

]

# Define trigger words for event extraction
trigger_words = [
    'announced', 'passed', 'introduced', 'unveiled', 'declared', 'debated', 'discussed', 'hearing',
    'reviewed', 'rejected', 'voted', 'implemented', 'enforced', 'issued', 'ruled', 'guideline',
    'regulation', 'challenged', 'appealed', 'protested', 'supported', 'poll', 'report', 'study',
    'signed', 'agreed', 'treaty', 'consultation', 'emergency', 'imposed', 'urgent', 'order', 'plan',
     'imposed', 'sit-in', 'striked', 'okayed', 'Formed', 'rallies'
     
]

# Compile the trigger words into a regular expression pattern for easier matching
trigger_pattern = re.compile(r'\b(?:' + '|'.join(trigger_words) + r')\b', re.IGNORECASE)

# Limit the number of sentences in the event description
MAX_EVENT_SENTENCES = 2
# Helper function to extract arguments using dependency parsing
def extract_arguments(doc):
    arguments = {
        "Actor": [],
        "Action": [],
        "Target": [],
        "Location": [],
        "Time": []
    }
    
    for token in doc:
        # Actor: Find subjects or named entities related to actions
        if token.dep_ in ("nsubj", "nsubjpass") and token.ent_type_ in ("PERSON", "ORG", "GPE"):
            arguments["Actor"].append(token.text)

        # Action: Verbs that are related to the trigger words
        if token.pos_ == "VERB" and trigger_pattern.search(token.text):
            arguments["Action"].append(token.lemma_)

        # Target: Direct objects of the action
        if token.dep_ == "dobj":
            arguments["Target"].append(token.text)

        # Location: Named entities labeled as GPE or LOC
        if token.ent_type_ in ("GPE", "LOC"):
            arguments["Location"].append(token.text)

        # Time: Named entities labeled as DATE or TIME
        if token.ent_type_ in ("DATE", "TIME"):
            arguments["Time"].append(token.text)

    # Remove duplicates
    for key in arguments:
        arguments[key] = list(set(arguments[key]))

    return arguments

# Create a CSV file to store the dataset
with open('enhanced_event_dataset.csv', mode='w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow(["Title", "Event Type", "Event Description", "Arguments", "Entities", "Publish Date", "URL"])

    # Process each URL with a delay
    for url in urls:
        article = Article(url)
        article.download()
        article.parse()

        # Split the article into sentences
        sentences = sent_tokenize(article.text)

        # Find sentences containing any of the event trigger words
        event_sentences = [sentence for sentence in sentences if trigger_pattern.search(sentence)]
        
        # Limit the event description to a maximum of MAX_EVENT_SENTENCES
        event_description = ' '.join(event_sentences[:MAX_EVENT_SENTENCES]) if event_sentences else 'No specific event mentioned'

        # Identify event types by finding trigger words in the text
        event_triggers = [word for word in word_tokenize(article.text.lower()) if word in trigger_words]
        event_type = ', '.join(set(event_triggers)) if event_triggers else 'None'

        # Process the article text with spaCy to extract named entities
        doc = nlp(article.text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Extract arguments
        arguments = extract_arguments(doc)

        # Check if the publish date is available; if not, try to extract it from the text
        publish_date = article.publish_date
        if not publish_date:
            # Attempt to find dates within the first few lines of the text
            date_entities = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
            publish_date = date_entities[0] if date_entities else "Unknown"

        # Write the data to the CSV file
        writer.writerow([
            article.title,
            event_type,
            event_description,
            arguments,
            entities,
            publish_date,
            article.url
        ])

        # Introduce a delay of 2 seconds between requests
        time.sleep(2)

print("Enhanced dataset collection complete. Data saved to 'enhanced_event_dataset.csv'.")  
