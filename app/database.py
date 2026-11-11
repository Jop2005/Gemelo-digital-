"""
MongoDB database connection for the Digital Twin backend
"""
from pymongo import MongoClient
from pymongo.database import Database
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/GemeloDigitalCujai")
MONGODB_DB = os.getenv("MONGODB_DB", "GemeloDigitalCujai")

# Global client instance
_client: MongoClient | None = None
_db: Database | None = None


def get_database() -> Database:
    """
    Get MongoDB database instance with connection pooling.
    """
    global _client, _db

    if _db is None:
        _client = MongoClient(MONGODB_URI)
        _db = _client[MONGODB_DB]
        print(f"✅ Connected to MongoDB: {MONGODB_DB}")

    return _db


def close_database():
    """
    Close MongoDB connection.
    """
    global _client, _db

    if _client:
        _client.close()
        _client = None
        _db = None
        print("❌ MongoDB connection closed")
