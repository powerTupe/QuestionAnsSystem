// install the allen using the commmand "!pip install allennlp==1.0.0 allennlp-models==1.0.0"//

from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bidaf-model-2020.03.19.tar.gz")

passage = """
Maharashtra, the most affected state overall, has reported 67,160 new cases to take its tally to 4228836. The state has added 650,676 cases in the past 10 days. Kerala, the second-most-affected state by total tally, has added 26,685 cases to take its tally to 1377186.
"""

# "what is GDP growth rate of india?"
# "how many political parties are there in india?"
# "how are union territoris managed?"
# "how many states are there in india?"

result=predictor.predict(
  passage=passage,
  question= "how much case in maharashtra?"
  # "how are union territoris managed?"
)
result['best_span_str']
