from flask import request
from flask_restplus import Namespace, Resource, fields
import random
from collections import OrderedDict

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
#from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
#from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn import utils
import pickle

filename = 'swapp_regression.sav'

namespace = Namespace('assessment', 'Endpoint submit a farm assessment and get a recommendation.')

countries = ['Ivory Coast', 'Ghana', 'Liberia', 'Nigeria', 'Sierra Leone']
yes_no_na = ['Yes', 'No', 'Not applicable']
bpm_ignore_list = ['female', 'ivory coast', 'ghana', 'liberia', 'nigeria', 'sierra leone', 'drainage_not applicable', 'integrated_pest_management_not applicable']

assessment_model = namespace.model('Assessment', {
    # Farmer input
    'gender': fields.String(required=True, description='Gender', enum=['Male', 'Female']),
    'country_name': fields.String(required=True, description='Country',
                                  enum=countries),

    # Farm input
    'yield_major': fields.Integer(required=True,
                                  description='What was the yield of the last MAJOR season (in metric tons)?'),
    'yield_minor': fields.Integer(required=True,
                                  description='What was the yield of the last MINOR season (in metric tons)?'),
    'size_ha': fields.Integer(required=True, description='What is the size of your farm (in hectares)?'),

    # BPM
    'clean_palm_circle': fields.Boolean(required=True, description='Do you have clean palm circles?'),
    'access_path': fields.Boolean(required=True, description='Do you have access paths on the farm?'),
    'cover_crop': fields.Boolean(required=True, description='Have you planted leguminous cover crop?'),
    'weed': fields.Boolean(required=True, description='Do you weed your farm?'),  # Custom
    'prune': fields.Boolean(required=True, description='Do you prune your oil palm trees?'),  # Custom
    'empty_prune_bunches': fields.Boolean(required=True, description='Do you usually apply empty fruit bunches?'),
    'drainage': fields.String(required=True,
                               description='Have you provided adequate drainage systems (esp in water logged areas)?', enum=yes_no_na),
    'box_frond_stacking': fields.Boolean(required=True, description='-Do you do Box Frond Stacking?'),
    'chisels': fields.Boolean(required=True, description='Do you have harvesting/pruning chisels/sickles for working?'),
    'apply_fertilizer': fields.Boolean(required=True, description='Do you apply Fertilizer?'),
    'apply_pesticides': fields.Boolean(required=True,
                                       description='Did you apply Pesticides to this your largest bearing plot LAST YEAR?'),
    'apply_herbicides': fields.Boolean(required=True,
                                       description='Did you apply Herbicides to this your largest bearing plot LAST YEAR?'),
    'bearing_fruit': fields.Boolean(required=True, description='Is this oil palm farm bearing fruit?'),
    'integrated_pest_management': fields.String(required=True, description='Do you have Integrated Pest Management?', enum=yes_no_na),
})

assessment_result_model = namespace.model('AssessmentResult', {
    'practice': fields.String(description='The practice to apply'),
  #  'current_yield': fields.Integer(description='The current yield for major season'),
    'expected_yield': fields.Integer(description='The expected yield for major season after applying this BPM')
})

assessment_result = namespace.model('AssessmentResultList', {
    'recommendations': fields.Nested(assessment_result_model, description='A list of BPM recommendations sorted by most impact.')
})


@namespace.route('')
class Assessment(Resource):

    @namespace.expect(assessment_model)
    @namespace.marshal_list_with(assessment_result_model)
    @namespace.response(500, 'Internal Server error')
    def post(self):
        '''Assessment endpoint'''

        data_cleaned = self.clean(request.json)
        print(data_cleaned)

        recommendations = self.predict(data_cleaned)
        print(recommendations)

        return recommendations

    def clean(self, data):
        print(data)
        input = OrderedDict([
            ('apply_pesticides', data['apply_pesticides']),
            ('apply_herbicides', data['apply_herbicides']),
            ('bearing_fruit', data['bearing_fruit']),

            #countries
            ('ivory coast', data['country_name'].lower() == 'ivory coast'),
            ('ghana', data['country_name'].lower() == 'ghana'),
            ('liberia', data['country_name'].lower() == 'liberia'),
            ('nigeria', data['country_name'].lower() == 'nigeria'),
            ('sierra leone', data['country_name'].lower() == 'sierra leone'),

            ('female', data['gender'].lower() == 'female'),
            ('clean_palm_yes', data['clean_palm_circle']),
            ('access_paths_yes', data['access_path']),
            ('cover_crop_yes', data['cover_crop']),
            ('empty_fruit_bunches_yes', data['empty_prune_bunches']),
            ('box_frond_stacking_yes', data['box_frond_stacking']),
            ('chisels_yes', data['chisels']),
            ('apply_fertilizer_yes', data['apply_fertilizer']),

            # ('yield', data['yield_major']), # yield_major
            ('weed', data['weed']),
            ('prune', data['prune']),
            ('drainage_not applicable', data['drainage'].lower() == 'not applicable'),
            ('drainage_yes', data['drainage'].lower() == 'yes'),
            ('integrated_pest_management_not applicable', data['integrated_pest_management'].lower() == 'not applicable'),
            ('integrated_pest_management_yes', data['integrated_pest_management'].lower() == 'yes'),

            ('size_ha', data['size_ha'])
        ])

        # Add countries
        for country in countries:
            cl = country.lower()
            dcn = data['country_name'].lower()
            input[cl] = dcn == cl

        return input

    def predict(self, input):
        bpm_improv = []

        # Create new prediction for each pbm that isn't applied yet
        bpm_improv_sorted = []
        for bpm, val in input.items():
            print(bpm, val)
            print(isinstance(val, bool), (bpm not in bpm_ignore_list), val)
            if not isinstance(val, bool) or (bpm in bpm_ignore_list) or val:
                continue

            # For now do a dumb skip check if drainage is not applicable
            if bpm == 'drainage_not applicable' and val:
                continue
            # For now do a dumb skip check if pest management is not applicable
            if bpm == 'integrated_pest_management_not applicable' and val:
                continue


            # Get model prediction
            result = round(self.ext_prediction(input))

            # Create a list with entries
            bpm_improv.append({
                'practice': bpm,
             #   'current_yield': input['yield'],
                'expected_yield': result
            })

            # Sort by higher expected yield first
            bpm_improv_sorted = sorted(bpm_improv, key=lambda d: d['expected_yield'], reverse=True)

        return bpm_improv_sorted

    def ext_prediction(self, input):
        input_array = [item[1] if not isinstance(item[1], bool) else int(item[1]) for item in input.items()]
        print(input_array)

        #return random.randrange(20, 400)
        return self.predict_output(input_array)


    def predict_output(self, data):
        an_array = np.array(data)

        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.predict(an_array.reshape(1, -1))
        print(result)
        return result[0]