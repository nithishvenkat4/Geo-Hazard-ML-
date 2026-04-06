import sys, os
sys.path.append(os.path.abspath("."))from flask import Flask

from apps.flask.routes import main
app = Flask(__name__)
app.register_blueprint(main)

if __name__ == "__main__":
    app.run(debug=True)