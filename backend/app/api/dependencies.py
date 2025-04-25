from fastapi import Depends, HTTPException, status
from app.services.general_classifier import GeneralClassifierService
from app.services.dog_classifier import DogClassifierService
from app.services.apolo_classifier import ApoloClassifierService

# Create instances of our services
general_classifier_service = GeneralClassifierService()
dog_classifier_service = DogClassifierService()
apolo_classifier_service = ApoloClassifierService()

# Dependency functions


def get_general_classifier_service():
    return general_classifier_service


def get_dog_classifier_service():
    return dog_classifier_service


def get_apolo_classifier_service():
    return apolo_classifier_service
