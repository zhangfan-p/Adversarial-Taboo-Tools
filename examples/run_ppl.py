from utils.judge_tools import SentencePPL
import json
from linecache import getline

def main():
    file_name = "reddit_example.json"
    cal_ppl = SentencePPL()
    for line in range(1,11):
        js = json.loads(getline(file_name, line))
        sent = js["post"]
        ppl = cal_ppl(sent)
        print("Sentence:", sent)
        print("Perplexity:", ppl, "\n")


if __name__ == "__main__":
    main()
