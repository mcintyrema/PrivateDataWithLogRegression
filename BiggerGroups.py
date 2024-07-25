from pyarxaas import ARXaaS
import pyarxaas.models
import pyarxaas.models.anonymization_metrics
import pyarxaas.models.anonymize_result
import pyarxaas.models.risk_profile
from pyarxaas.privacy_models import KAnonymity
from pyarxaas.hierarchy import IntervalHierarchyBuilder
from pyarxaas import AttributeType
from pyarxaas import Dataset
import pandas as pd
import requests


arxaas = ARXaaS('http://localhost:8081') 

#load data into dataset
df = pd.read_csv("../adult.csv")

# create dataset
dataset = Dataset.from_pandas(df)

#select quasi identifiers and insensitive types
dataset.set_attribute_type(AttributeType.INSENSITIVE, "workclass", "fnlwgt", "education", "marital.status", "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss", "hours.per.week", "native.country")
dataset.set_attribute_type(AttributeType.QUASIIDENTIFYING, 'age', 'education.num')
dataset.set_attribute_type(AttributeType.IDENTIFYING, 'income')

# get the risk profle of the dataset before anonymizing
risk_profile = arxaas.risk_profile(dataset)
privacy_risk = risk_profile.re_identification_risk

# build age hierarchy
age_hierarchy_builder = IntervalHierarchyBuilder()
age_hierarchy_builder.add_interval(-1, 0, "unknown")
age_hierarchy_builder.add_interval(0, 18, "child")
age_hierarchy_builder.add_interval(18, 30, "young-adult")
age_hierarchy_builder.add_interval(30, 75, "adult")
age_hierarchy_builder.add_interval(75, 110, "elderly")
age_hierarchy = arxaas.hierarchy(age_hierarchy_builder, df['age'].tolist())

# build education hierarchy
education_hierarchy_builder = IntervalHierarchyBuilder()
education_hierarchy_builder.add_interval(-1, 0, "unknown")
education_hierarchy_builder.add_interval(0, 9, "Some school")
education_hierarchy_builder.add_interval(9, 10, "HS-grad")
education_hierarchy_builder.add_interval(10, 23, "Higher Educatoin")
education_hierarchy = arxaas.hierarchy(education_hierarchy_builder, df['education.num'].tolist())

#add hierarchies
dataset.set_hierarchy('age', age_hierarchy)
dataset.set_hierarchy('education.num', education_hierarchy)

#create privacy model
kanon = KAnonymity(2)

anon_result = arxaas.anonymize(dataset, [kanon])
#verify anonymization
# print(anon_result.anonymization_status)

risk_profile = arxaas.risk_profile(dataset)
privacy_risk = risk_profile.re_identification_risk
# print(privacy_risk)