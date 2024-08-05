import os
import json
import random
import pandas as pd

with open("../allocation_tasks/config.json", "r") as f:
    config = json.load(f)
essay_rating_file = os.path.join("allocation_tasks", config["essay_rating_file"])

class CandidateSampler:
    def __init__(self, candidates):
        random.shuffle(candidates)
        self.candidates = candidates
        self.size = len(candidates)
        self.index = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.index > (self.size - 1):
            random.shuffle(self.candidates)
            self.index = 0
        candidate = self.candidates[self.index]
        self.index += 1
        return candidate
    
    
def load_essay_rating():
    rating_df = pd.read_excel(essay_rating_file, sheet_name="4_Summary", skiprows=3)
    rating_df = rating_df[rating_df["S/E"] == "E"]
    rating_df = rating_df.iloc[:, :19]
    rating_df["student_id"] = rating_df["Original Code"].apply(lambda x: x.split("_")[2])
    rating_df = rating_df.rename(columns={"Overall Rating Score (/100)": "Human Rating"})
    rating_df = rating_df[["Region", "student_id", "L2 Proficiency", "Human Rating"]]
    rating_df = rating_df.rename(columns={"Region": "group"})
    bins = [0, 40, 53, 100]
    rating_df["label"] = pd.cut(rating_df["Human Rating"], bins=bins, labels=["Fair", "Average", "Good"])
    return rating_df