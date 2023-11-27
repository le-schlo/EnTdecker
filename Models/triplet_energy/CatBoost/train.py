from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.DataImport.Module.CSV import CSV
from EasyChemML.Encoder.MFF import MFF
from EasyChemML.Encoder.impl_RdkitConverter.MolRdkitConverter import MolRdkitConverter
from EasyChemML.Environment import Environment
from EasyChemML.JobSystem.JobFactory.JobFactory import Job_Factory
from EasyChemML.JobSystem.JobFactory.Module.Jobs import ModelTrainJob, ModelTrainEvalJob, ModelPredictJob
from EasyChemML.JobSystem.Utilities.Config import Config
from EasyChemML.Metrik.MetricStack import MetricStack
from EasyChemML.Metrik.Module.MeanAbsoluteError import MeanAbsoluteError
from EasyChemML.Metrik.Module.MeanSquaredError import MeanSquaredError
from EasyChemML.Metrik.Module.R2_Score import R2_Score
from EasyChemML.Model.CatBoost_r import CatBoost_r
from EasyChemML.JobSystem.Runner.Module.LocalRunner import LocalRunner
from EasyChemML.Splitter.Module.ShuffleSplitter import ShuffleSplitter
from EasyChemML.Splitter.Splitcreator import Splitcreator
from EasyChemML.Utilities.Dataset import Dataset
import os

# ----------------------------------- Data Preprocessing -----------------------

os.environ["HDF5_USE_FILE_LOCKING"] = 'False'
env = Environment(WORKING_path_addRelativ='Output')
step_size = 100000
threads = 1
print('START')
dataset_name = 'EnTdecker_dataset'
dataLoader = {('%s' % dataset_name): CSV('/tmp/EnTdecker/Data/Retrain_Disulfides.csv')}
di = DataImporter(env)
bp = di.load_data_InNewBatchPartition(dataLoader, max_chunksize=100000)

print('Start MolRdkitConverter')
molRdkit_converter = MolRdkitConverter()
molRdkit_converter.convert(bp[dataset_name], columns=['smiles'], n_jobs=4)

mff_encoder = MFF()
mff_encoder.convert(datatable=bp[dataset_name], columns=['smiles'], fp_length=2048, n_jobs=64)



split_creator = Splitcreator()
splitter_EnTdecker = ShuffleSplitter(1, 42, test_size=0.15)
splitset_EnTdecker = split_creator.generate_split(bp[dataset_name], splitter_EnTdecker)

dataset_EnTdecker = Dataset(bp[dataset_name],
                            name=dataset_name,
                            feature_col=['smiles'],
                            target_col=['e_t'],
                            split=splitset_EnTdecker, env=env)


# ----------------------------------- Training --------------------------------------

job_factory = Job_Factory(env)
job_runner = LocalRunner(env)

r2score = R2_Score()
mae = MeanAbsoluteError()
rmse = MeanSquaredError()
metricStack_r = MetricStack({'r2': r2score, 'mae': mae, 'rmse': rmse})

catboost_r = Config(
    CatBoost_r,
    {'verbose': 100,
     'thread_count': 64,
     'allow_writing_files': True,
     'iterations': 10000,
     'depth': 10}
)

job: ModelTrainEvalJob = job_factory.create_ModelTrainEvalJob(
    'CatBoost_EnTdecker',
    dataset_EnTdecker,
    catboost_r,
    metricStack_r,
    dataset_EnTdecker.get_Splitset().get_outer_split(0)
)

job_runner.run_Job(job)

# ----------------------------------- Prediction ----------------------------------

print(f'Train_Metrics: {job.result_metric_TRAIN}')
print(f'Test_Metrics: {job.result_metric_TEST}')

job.trained_Model.save_model('/tmp/EnTdecker/Models/triplet_energy/CatBoost/model.catb')
env.clean()
