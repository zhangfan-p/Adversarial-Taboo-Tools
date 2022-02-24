from utils.judge_tools import SentenceMatch
import json
from linecache import getline

def main():
    file_name = "reddit_example.json"
    match_predict = SentenceMatch()
    for line in range(1,6):
        js = json.loads(getline(file_name, line))
        predict_label = match_predict(js["post"], js["response"])
        print("The predict label is: ", predict_label)
        print("The real label is: ", js["label"], "\n")


if __name__ == "__main__":
    main()