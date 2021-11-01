from flask import Blueprint
from flask_restplus import Api
from endpoints.assessment import namespace as ns_assessment

blueprint = Blueprint('documented_api', __name__, url_prefix='/api/v1')

api_extension = Api(
    blueprint,
    title='Assessment',
    version='0.1',
    description='Do a farm assessment',
    doc='/doc'
)

api_extension.add_namespace(ns_assessment)