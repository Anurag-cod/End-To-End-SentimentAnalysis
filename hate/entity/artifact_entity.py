from dataclasses import dataclass

# Data ingestion artifacts (notebook flow: training + validation paths)
@dataclass
class DataIngestionArtifacts:
    training_file_path: str
    validation_file_path: str




@dataclass
class DataTransformationArtifacts:
    transformed_data_path: str




@dataclass
class ModelTrainerArtifacts: 
    trained_model_path:str
    x_test_path: list
    y_test_path: list



@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool 



@dataclass
class ModelPusherArtifacts:
    bucket_name: str

