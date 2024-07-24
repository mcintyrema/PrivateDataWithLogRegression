from pyarxaas import ARXaaS
from pyarxaas.privacy_models import KAnonymity
from pyarxaas import AttributeType
from pyarxaas import Dataset
import pandas as pd

arxaas = ARXaaS(url) # url contains url to AaaS web service

df = pd.read_csv("../data/adult.csv/adult.csv")

# create Dataset
dataset = Dataset.from_pandas(df)


# set attribute type
dataset.set_attributes(AttributeType.QUASIIDENTIFYING, 'name', 'gender')
dataset.set_attribute(AttributeType.IDENTIFYING, 'id')

# get the risk profle of the dataset
risk_profile = arxaas.risk_profile(dataset)

# get risk metrics
re_indentifiation_risk = risk_profile.re_identification_risk
distribution_of_risk = risk_profile.distribution_of_risk