import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw data — hardcoded as dataset must be downloaded manually from Kaggle
RAW_TRANSACTION_PATH = os.path.join(BASE_DIR, "ieee-fraud-detection", "train_transaction.csv")
RAW_IDENTITY_PATH    = os.path.join(BASE_DIR, "ieee-fraud-detection", "train_identity.csv")
TEST_TRANSACTION_PATH = os.path.join(BASE_DIR, "ieee-fraud-detection", "test_transaction.csv")
TEST_IDENTITY_PATH    = os.path.join(BASE_DIR, "ieee-fraud-detection", "test_identity.csv")

aws_uri = "http://ec2-35-159-196-177.eu-central-1.compute.amazonaws.com:5000"
