from pyarxaas import ARXaaS
from pyarxaas.privacy_models import KAnonymity
from pyarxaas.hierarchy import IntervalHierarchyBuilder
from pyarxaas import AttributeType
from pyarxaas import Dataset
import pandas as pd


arxaas = ARXaaS('http://localhost:8081') 

#load data into dataset
df = pd.read_csv("../adult.csv")
dataset = Dataset.from_pandas(df)

#select quasi identifiers
dataset.set_attribute_type(AttributeType.QUASIIDENTIFYING, 'age', 'education.num')
dataset.set_attribute_type(AttributeType.IDENTIFYING, 'income')

# build age hierarchy
age_hierarchy_builder = IntervalHierarchyBuilder()
age_hierarchy_builder.add_interval(-1, 0, "unknown")
age_hierarchy_builder.add_interval(0, 18, "child")
age_hierarchy_builder.add_interval(18, 30, "young-adult")
age_hierarchy_builder.add_interval(30, 60, "adult")
age_hierarchy_builder.add_interval(60, 75, "older adult")
age_hierarchy_builder.add_interval(75, 110, "elderly")
age_hierarchy = arxaas.hierarchy(age_hierarchy_builder, df['age'].tolist())

# build education hierarchy
education_hierarchy_builder = IntervalHierarchyBuilder()
education_hierarchy_builder.add_interval(-1, 0, "unknown")
education_hierarchy_builder.add_interval(0, 8, "Some school")
education_hierarchy_builder.add_interval(8, 9, "HS-grad")
education_hierarchy_builder.add_interval(9, 13, "Undergrad")
education_hierarchy_builder.add_interval(13, 23, "Grad School")
education_hierarchy = arxaas.hierarchy(education_hierarchy_builder, df['education.num'].tolist())

# get the risk profle of the dataset before anonymizing
risk_profile = arxaas.risk_profile(dataset)
privacy_risk = risk_profile.re_identification_risk

#add hierarchies
dataset.set_hierarchy('age', age_hierarchy)
dataset.set_hierarchy('education.num', education_hierarchy)

#create privacy model
kanon = KAnonymity(2)
# Try anonymizing the dataset
anon_result = arxaas.anonymize(dataset, [kanon])
print(anon_result.anonymization_status)

# anon_result.dataset.to_dataframe()

# get risk after anonymizing

# print(anon_result.anonymization_status)