import json
import random
import pandas as pd
import os
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=False, default="./data/source_data/meta_character.json",
                    help="Directory where the character JSON file is located")
args = parser.parse_args()

def make_cross_qa_df(characters):

    templates = [
            "Do you know {name}?",
            "Have you heard of {name}?",
            "Have you met {name}?"
        ]
    
    df = pd.DataFrame(columns=['country','character', 'question', 'answer'])

    rows = []
    for country in characters:
        for char in characters[country]:
            for other in characters[country]:
                if other["name"] == char["name"]:
                    continue
                if other["time"] == "present" and char["time"]=="past":
                    q = random.choice(templates).format(name=other["name"])
                    a = "I can not answer that question."

                    rows.append({
                        "country" : country,
                        "character" : char["name"],
                        "profile" : char["profile"],
                        "question": q,
                        "answer": a
                    })
    
    if rows:
        df = pd.DataFrame(rows)

    return df



if __name__ == "__main__":

    with open(args.input_dir, 'r', encoding='utf-8') as f:
        characters = json.load(f)

    result_df = make_cross_qa_df(characters=characters)

    output_dir = f'{args.input_dir.split('meta_character')[0]}cross_universe_qa.csv'

    result_df.to_csv(output_dir, encoding='utf-8', index=False)
    