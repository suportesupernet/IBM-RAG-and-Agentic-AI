def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

model_id = 'meta-llama/llama-3-3-70b-instruct'

parameters = {
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.2,
}

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
    # "api_key": "your api key"
}

project_id = "skills-network"

model ModelInference(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

msg = model.generate("In today's sales meeting, we ")
print(msg['results'][0]['generated_text'])