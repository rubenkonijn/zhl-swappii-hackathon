from flask import Flask
from endpoints import blueprint as assessment_endpoint

app = Flask('zhl')
app.config['RESTPLUS_MASK_SWAGGER'] = False
app.config['DEBUG'] = True

app.register_blueprint(assessment_endpoint)

if __name__ == "__main__":
    app.run()

